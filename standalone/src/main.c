/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <console/console.h>
#include <drivers/gpio.h>
#include <drivers/uart.h>
#include <kernel.h>
#include <power/reboot.h>
#include <stdio.h>
#include <sys/printk.h>
#include <sys/ring_buffer.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/crt.h>
#include <tvm/runtime/crt/graph_runtime.h>
#include <tvm/runtime/crt/packed_func.h>
#include <unistd.h>
#include <zephyr.h>

#ifdef CONFIG_ARCH_POSIX
#include "posix_board_if.h"
#endif

#include "crt_config.h"

#include "../inputs.c.inc"
#include "../graph_json.c.inc"

K_SEM_DEFINE(tx_sem, 0, 1);

static const struct device* tvm_uart;

int write_hook(int c) {
  uart_poll_out(tvm_uart, c);
  return 0;
}

/* ssize_t write_serial(void* unused_context, const uint8_t* data, size_t size) { */
/*   for (size_t i = 0; i < size; i++) { */
/*     uart_poll_out(tvm_uart, data[i]); */
/*   } */

/*   return size; */
/* } */

size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return vsnprintk(out_buf, out_buf_size_bytes, fmt, args);
}

void TVMPlatformAbort(tvm_crt_error_t error) {
  printf("TVMPlatformAbort: %08x\n", error);
  sys_reboot(SYS_REBOOT_COLD);
  for (;;)
    ;
}

int mem_pool_counts[10] = {0};

/* K_HEAP_DEFINE(tvm_heap, 200 * 1024); */

/* tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLContext ctx, void** out_ptr) { */
/*   *out_ptr = k_heap_alloc(&tvm_heap, num_bytes, K_NO_WAIT); */
/*   return (*out_ptr == NULL) ? kTvmErrorPlatformNoMemory : kTvmErrorNoError; */
/* } */

/* tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLContext ctx) { */
/*   k_heap_free(&tvm_heap, ptr); */
/*   return kTvmErrorNoError; */
/* } */

K_MEM_POOL_DEFINE(small_memory_pool, 16, 16, (90 * 1024 / 16), 4);

K_MEM_POOL_DEFINE(mid_memory_pool, 128, 128, (30 * 1024 / 128), 4);

K_MEM_POOL_DEFINE(tvm_memory_pool, 1024, 1024, 230, 4);

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLContext ctx, void** out_ptr) {
  *out_ptr = NULL;
  if (num_bytes < 128) {
    *out_ptr = k_mem_pool_malloc(&small_memory_pool, num_bytes);
  }
  if (*out_ptr == NULL && num_bytes < 1024) {
    *out_ptr = k_mem_pool_malloc(&mid_memory_pool, num_bytes);
  }
  if (*out_ptr == NULL) {
    *out_ptr = k_mem_pool_malloc(&tvm_memory_pool, num_bytes);
  }
  return (*out_ptr == NULL) ? kTvmErrorPlatformNoMemory : kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLContext ctx) {
  k_free(ptr);
  return kTvmErrorNoError;
}

uint32_t g_utvm_start_time;

#define MILLIS_TIL_EXPIRY 200
#define TIME_TIL_EXPIRY (K_MSEC(MILLIS_TIL_EXPIRY))
K_TIMER_DEFINE(g_utvm_timer, /* expiry func */ NULL, /* stop func */ NULL);

int g_utvm_timer_running = 0;

#ifdef CONFIG_LED
/* The devicetree node identifier for the "led0" alias. */
#define LED0_NODE DT_ALIAS(led0)

#define LED0 DT_GPIO_LABEL(LED0_NODE, gpios)
#define PIN DT_GPIO_PIN(LED0_NODE, gpios)
#define FLAGS DT_GPIO_FLAGS(LED0_NODE, gpios)

static struct device* led_pin;
#endif  // CONFIG_LED

int TVMPlatformTimerStart() { return -1; }

int TVMPlatformTimerStop(double* elapsed_time_seconds) { return -1; }

#define RING_BUF_SIZE 512
struct uart_rx_buf_t {
  struct ring_buf buf;
  uint32_t buffer[RING_BUF_SIZE];
};

struct uart_rx_buf_t uart_rx_buf;

void uart_irq_cb(const struct device* dev, void* user_data) {
  while (uart_irq_update(dev) && uart_irq_is_pending(dev)) {
    struct uart_rx_buf_t* buf = (struct uart_rx_buf_t*)user_data;
    if (uart_irq_rx_ready(dev) == 0) {
      continue;
    }

    uint8_t data[32];
    for (;;) {
      int bytes_read = uart_fifo_read(dev, data, sizeof(data));
      if (bytes_read < 0) {
        TVMPlatformAbort(0xbeef);
      } else if (bytes_read == 0) {
        break;
      }
      int bytes_written = ring_buf_put(&buf->buf, data, bytes_read);
      CHECK_EQ(bytes_read, bytes_written, "bytes_read: %d; bytes_written: %d", bytes_read,
               bytes_written);
    }
  }
}

void uart_rx_init(struct uart_rx_buf_t* buf, const struct device* dev) {
  ring_buf_init(&buf->buf, RING_BUF_SIZE, buf->buffer);
  uart_irq_callback_user_data_set(dev, uart_irq_cb, (void*)buf);
  uart_irq_rx_enable(dev);
}

int uart_rx_buf_read(struct uart_rx_buf_t* buf, uint8_t* data, size_t data_size_bytes) {
  unsigned int key = irq_lock();
  int bytes_read = ring_buf_get(&buf->buf, data, data_size_bytes);
  irq_unlock(key);
  return bytes_read;
}

/*! \brief macro to do C API call */
#define TVM_CCALL(func)                                                              \
  do {                                                                               \
    tvm_crt_error_t ret = (func);                                                    \
    if (ret != kTvmErrorNoError) {                                                   \
      fprintf(stderr, "%s: %d: error: %s\n", __FILE__, __LINE__, TVMGetLastError()); \
      exit(ret);                                                                     \
    }                                                                                \
  } while (0)

TVMModuleHandle TVMArgs_AsModuleHandle(const TVMArgs* args, size_t index);

extern void __stdout_hook_install(int (*hook)(int));
void main(void) {
  /* Claim console device */
//  tvm_uart = device_get_binding(DT_LABEL(DT_CHOSEN(zephyr_console)));
//  uart_rx_init(&uart_rx_buf, tvm_uart);
//  __stdout_hook_install(&write_hook);
  printk("uTVM Standalone Demo\n");

  int64_t device_type = kDLCPU;
  int64_t device_id = 0;

  TVMContext ctx;
  ctx.device_type = (DLDeviceType)device_type;
  ctx.device_id = device_id;
  TVM_CCALL(TVMInitializeRuntime());

  TVMPackedFunc pf;
  TVMArgs args = TVMArgs_Create(NULL, NULL, 0);
  TVM_CCALL(TVMPackedFunc_InitGlobalFunc(&pf, "runtime.SystemLib", &args));
  TVM_CCALL(TVMPackedFunc_Call(&pf));

  TVMModuleHandle mod_syslib = TVMArgs_AsModuleHandle(&pf.ret_value, 0);

  TVMGraphRuntime* graph_runtime = NULL;
  TVM_CCALL(TVMGraphRuntime_Create(graph_json, mod_syslib, &ctx, &graph_runtime));
  TVMGraphRuntime_SetInput(graph_runtime, "data", (DLTensor*) &input_data_tensor);
  TVMGraphRuntime_Run(graph_runtime);

  int8_t data[10];
  int64_t shape[2] = {1, 10};
  DLTensor out = {&data, {kDLCPU, 0}, 2, {kDLInt, 8, 0}, shape, NULL, 0};
  TVM_CCALL(TVMGraphRuntime_GetOutput(graph_runtime, 0, &out));
  printf("TVM complete! Output: ");
  for (int i = 0; i < out.shape[1]; i++) {
    if (i > 0) {
      printf(", ");
    }
    printf("%d", ((int8_t*) out.data)[i]);
  }
  printf("\n");
#ifdef CONFIG_ARCH_POSIX
  posix_exit(0);
#else
  for (;;) ;
#endif
}
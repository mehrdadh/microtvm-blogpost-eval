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

#include <drivers/gpio.h>
#include <drivers/uart.h>
#include <fatal.h>
#include <kernel.h>
#include <power/reboot.h>
#include <stdio.h>
#include <sys/printk.h>
#include <sys/ring_buffer.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/utvm_rpc_server.h>
#include <unistd.h>
#include <zephyr.h>
#include <random/rand32.h>

#ifdef CONFIG_ARCH_POSIX
#include "posix_board_if.h"
#endif

#include "crt_config.h"

K_SEM_DEFINE(tx_sem, 0, 1);

static const struct device* tvm_uart;

int write_hook(int c) {
  uart_poll_out(tvm_uart, c);
  return 0;
}

#ifdef CONFIG_LED
/* The devicetree node identifier for the "led0" alias. */
#define LED0_NODE DT_ALIAS(led0)

//#define LED0 "GPIOF"
//#define PIN 12
//#define FLAGS 0

#define LED0 DT_GPIO_LABEL(LED0_NODE, gpios)
#define PIN DT_GPIO_PIN(LED0_NODE, gpios)
#define FLAGS DT_GPIO_FLAGS(LED0_NODE, gpios)
static const struct device* led_pin;
#endif  // CONFIG_LED

static size_t g_num_bytes_requested = 0;
static size_t g_num_bytes_written = 0;

static const uint8_t* g_transmit_data = NULL;
static size_t g_transmit_data_size = 0;
static volatile size_t g_transmitted_bytes = 0;
static volatile bool g_transmit_complete = true;

ssize_t write_serial(void* unused_context, const uint8_t* data, size_t size) {
#ifdef CONFIG_LED
  gpio_pin_set(led_pin, PIN, 1);
#endif

  g_num_bytes_requested += size;

 for (size_t i = 0; i < size; i++) {
   uart_poll_out(tvm_uart, data[i]);
   g_num_bytes_written++;
 }

/*   if (size > 0) { */
/*     g_transmit_data = data; */
/*     g_transmit_data_size = size; */
/* //    g_transmitted_bytes = 1; */
/* //    uart_poll_out(tvm_uart, data[0]); */
/*     g_transmitted_bytes = 0; */
/* //    if (size > 1) { */
/*       g_transmit_complete = false; */
/*       uart_irq_tx_enable(tvm_uart); */
/*       while (!g_transmit_complete) {} */
/* //    } */
/*  } */

#ifdef CONFIG_LED
  gpio_pin_set(led_pin, PIN, 0);
#endif

  return size;
}

#if 0  // XXX MDW - DO NOT SUBMIT
void k_sys_fatal_error_handler(unsigned int reason, const z_arch_esf_t *esf) {
  gpio_pin_set(led_pin, PIN, 1);
  for (;;) ;
}
#endif

size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes,
                                const char* fmt, va_list args) {
  return vsnprintk(out_buf, out_buf_size_bytes, fmt, args);
}

void TVMPlatformAbort(tvm_crt_error_t error) {
  sys_reboot(SYS_REBOOT_COLD);
  for (;;)
    ;
}

uint32_t g_utvm_start_time;

#define MILLIS_TIL_EXPIRY 200
#define TIME_TIL_EXPIRY (K_MSEC(MILLIS_TIL_EXPIRY))
K_TIMER_DEFINE(g_utvm_timer, /* expiry func */ NULL, /* stop func */ NULL);

int g_utvm_timer_running = 0;

tvm_crt_error_t TVMPlatformTimerStart() {
  if (g_utvm_timer_running) {
    TVMLogf("timer already running");
    return kTvmErrorSystemErrorMask | 1;
  }

#ifdef CONFIG_LED
  gpio_pin_set(led_pin, PIN, 1);
#endif
  k_timer_start(&g_utvm_timer, TIME_TIL_EXPIRY, TIME_TIL_EXPIRY);
  g_utvm_start_time = k_cycle_get_32();
  g_utvm_timer_running = 1;
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformTimerStop(double* res_us) {
  if (!g_utvm_timer_running) {
    TVMLogf("timer not running");
    return kTvmErrorSystemErrorMask | 2;
  }

  uint32_t stop_time = k_cycle_get_32();
#ifdef CONFIG_LED
  gpio_pin_set(led_pin, PIN, 0);
#endif

  // compute how long the work took
  uint32_t cycles_spent = stop_time - g_utvm_start_time;
  if (stop_time < g_utvm_start_time) {
    // we rolled over *at least* once, so correct the rollover it was *only*
    // once, because we might still use this result
    cycles_spent = ~((uint32_t)0) - (g_utvm_start_time - stop_time);
  }

  uint32_t ns_spent = (uint32_t)k_cyc_to_ns_floor64(cycles_spent);
  double hw_clock_res_us = ns_spent / 1000.0;

  // need to grab time remaining *before* stopping. when stopped, this function
  // always returns 0.
  int32_t time_remaining_ms = k_timer_remaining_get(&g_utvm_timer);
  k_timer_stop(&g_utvm_timer);
  // check *after* stopping to prevent extra expiries on the happy path
  if (time_remaining_ms < 0) {
    TVMLogf("negative time remaining");
    return kTvmErrorSystemErrorMask | 3;
  }
  uint32_t num_expiries = k_timer_status_get(&g_utvm_timer);
  uint32_t timer_res_ms = ((num_expiries * MILLIS_TIL_EXPIRY) + time_remaining_ms);
  double approx_num_cycles =
      (double)k_ticks_to_cyc_floor32(1) * (double)k_ms_to_ticks_ceil32(timer_res_ms);
  // if we approach the limits of the HW clock datatype (uint32_t), use the
  // coarse-grained timer result instead
  if (approx_num_cycles > (0.5 * (~((uint32_t)0)))) {
    *res_us = timer_res_ms * 1000.0;
  } else {
    *res_us = hw_clock_res_us;
  }

  g_utvm_timer_running = 0;
  return kTvmErrorNoError;
}

K_MEM_POOL_DEFINE(tvm_memory_pool, 64, 1024, 216, 4);

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLContext ctx, void** out_ptr) {
  *out_ptr = k_mem_pool_malloc(&tvm_memory_pool, num_bytes);
  return (*out_ptr == NULL) ? kTvmErrorPlatformNoMemory : kTvmErrorNoError;
}

/*   // Align to 32-bits. */
/*   uintptr_t ptr = (uintptr_t) k_mem_pool_malloc(&tvm_memory_pool, num_bytes + sizeof(uintptr_t)); */
/*   if (ptr == 0) { */
/*     return kTvmErrorPlatformNoMemory; */
/*   } */

/*   int align_bytes = sizeof(uintptr_t) - (ptr & (sizeof(uintptr_t) - 1)); */
/*   ((uint8_t*) ptr)[align_bytes - 1] = align_bytes; */

/*   *out_ptr = (void*) (ptr + align_bytes); */
/*   return kTvmErrorNoError; */
/* } */

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLContext ctx) {
  k_free(ptr);
//  uint8_t align_bytes = *(((uint8_t*) ptr) - 1);
//  k_mem_pool_free((void*) (((uint8_t*) ptr) - align_bytes));
  return kTvmErrorNoError;
}

#define RING_BUF_SIZE 512
struct uart_rx_buf_t {
  struct ring_buf buf;
  uint32_t buffer[RING_BUF_SIZE];
};

struct uart_rx_buf_t uart_rx_buf;

void uart_irq_cb(const struct device* dev, void* user_data) {
  while (uart_irq_update(dev) && uart_irq_is_pending(dev)) {
    struct uart_rx_buf_t* buf = (struct uart_rx_buf_t*)user_data;
    if (uart_irq_rx_ready(dev) != 0) {
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
    /* if (!g_transmit_complete && uart_irq_tx_ready(dev) != 0) { */
    /*   size_t transmitted_bytes = g_transmitted_bytes; */
    /*   size_t to_transmit = g_transmit_data_size - transmitted_bytes; */
    /*   if (to_transmit != 0) { */
    /*     g_transmitted_bytes += uart_fifo_fill(tvm_uart, &g_transmit_data[transmitted_bytes], to_transmit); */
    /*   } else { */
    /*     uart_irq_tx_disable(tvm_uart); */
    /*     g_transmit_complete = true; */
    /*   } */
    /* } */
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

extern void __stdout_hook_install(int (*hook)(int));
void main(void) {
#ifdef CONFIG_LED
  led_pin = device_get_binding(LED0);
  if (led_pin == NULL) {
    for (;;)
      ;
  }
  int ret = gpio_pin_configure(led_pin, PIN, GPIO_OUTPUT_ACTIVE | FLAGS);
  if (ret < 0) {
    for (;;)
      ;
  }
  gpio_pin_set(led_pin, PIN, 1);
#endif

  /* Claim console device */
  tvm_uart = device_get_binding(DT_LABEL(DT_CHOSEN(zephyr_console)));
  const struct device* shadow_tvm_uart = tvm_uart;
  uart_rx_init(&uart_rx_buf, tvm_uart);

  utvm_rpc_server_t server = UTvmRpcServerInit(write_serial, NULL);
  TVMLogf("uTVM On-Device Runtime");
  gpio_pin_set(led_pin, PIN, 0);
  while (true) {
    uint8_t buf[256];
    int bytes_read = uart_rx_buf_read(&uart_rx_buf, buf, sizeof(buf));
    if (bytes_read > 0) {
      size_t bytes_remaining = bytes_read;
      uint8_t* cursor = buf;
      while (bytes_remaining > 0) {
        tvm_crt_error_t err = UTvmRpcServerLoop(server, &cursor, &bytes_remaining);
        if (err != kTvmErrorNoError && err != kTvmErrorFramingShortPacket) {
          TVMPlatformAbort(err);
        }
       if (g_num_bytes_written != 0 || g_num_bytes_requested != 0) {
          if (g_num_bytes_written != g_num_bytes_requested) {
            TVMPlatformAbort((tvm_crt_error_t) 0xbeef3);
          }
          g_num_bytes_written = 0;
          g_num_bytes_requested = 0;
        }
      }
    }
  }

#ifdef CONFIG_ARCH_POSIX
  posix_exit(0);
#endif
}

tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  sys_rand_get(buffer, num_bytes);
  return kTvmErrorNoError;
}

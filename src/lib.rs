use std::{iter, sync::{Arc, Mutex}};
use num_complex::{Complex};
use rustfft::{FftPlanner, num_traits::{Zero}};

pub mod dtype;
use crate::dtype::{ChunkedBuffer, RingBuffer};

pub trait StereoFilter {
    fn clear(&mut self);
    fn compute(&mut self, signal: (f64, f64)) -> (f64, f64);
}

pub struct TrueStereoFFTConvolution {
    ll: FFTConvolution,
    rr: FFTConvolution,
    lr: FFTConvolution,
    rl: FFTConvolution,
}
impl TrueStereoFFTConvolution {
    pub fn new(ir_ll: Vec<f64>, ir_rr: Vec<f64>, ir_lr: Vec<f64>, ir_rl: Vec<f64>, window_size: usize) -> TrueStereoFFTConvolution {
        TrueStereoFFTConvolution {
            ll: FFTConvolution::new(ir_ll, window_size),
            rr: FFTConvolution::new(ir_rr, window_size),
            lr: FFTConvolution::new(ir_lr, window_size),
            rl: FFTConvolution::new(ir_rl, window_size)
        }
    }
    pub fn window_size(&self) -> usize {
        self.ll.window_size()
    }
    pub fn internal_buffer_size(&self) -> usize {
        self.ll.internal_buffer_size()
    }
}
impl StereoFilter for TrueStereoFFTConvolution {
    fn clear(&mut self) {
        self.ll.clear();
        self.rr.clear();
        self.lr.clear();
        self.rl.clear();
    }
    fn compute(&mut self, signal: (f64, f64)) -> (f64, f64) {
        (self.ll.compute(signal.0) + self.rl.compute(signal.1),
        self.rr.compute(signal.1) + self.lr.compute(signal.0))
    }
}

pub struct StereoFFTConvolution {
    ll: FFTConvolution,
    rr: FFTConvolution,
}
impl StereoFFTConvolution {
    pub fn new(ir_left: Vec<f64>, ir_right: Vec<f64>, window_size: usize) -> StereoFFTConvolution {
        StereoFFTConvolution {
            ll: FFTConvolution::new(ir_left, window_size),
            rr: FFTConvolution::new(ir_right, window_size)
        }
    }
    pub fn window_size(&self) -> usize {
        self.ll.window_size()
    }
    pub fn internal_buffer_size(&self) -> usize {
        self.ll.internal_buffer_size()
    }
}
impl StereoFilter for StereoFFTConvolution {
    fn clear(&mut self) {
        self.ll.clear();
        self.rr.clear();
    }
    fn compute(&mut self, signal: (f64, f64)) -> (f64, f64) {
        (self.ll.compute(signal.0), self.rr.compute(signal.1))
    }
}

pub trait Filter {
    fn clear(&mut self);
    fn compute(&mut self, signal: f64) -> f64;
}

pub struct FFTConvolution {
    x: RingBuffer<Complex<f64>>,
    out: RingBuffer<f64>,
    window_size: usize,
    ir: Vec<f64>,
    ir_fft_cache: Vec<Complex<f64>>,
    fft_planner: Arc<Mutex<FftPlanner<f64>>>,
}

impl FFTConvolution {
    pub fn new(ir: Vec<f64>, window_size: usize) -> FFTConvolution {
        let padded_window_size = Self::padded_window_size(ir.len(), window_size);
        let mut ir_fft_cache: Vec<Complex<f64>> = ir.iter().map(|sample| Complex::new(*sample, 0.0)).chain(iter::repeat(Complex::zero()).take(padded_window_size - ir.len())).collect();
        let fft_planner = Arc::new(Mutex::new(FftPlanner::new()));
        {
            let fft = fft_planner.lock().unwrap().plan_fft_forward(padded_window_size);
            fft.process(&mut ir_fft_cache);
        }
        FFTConvolution {
            x: RingBuffer::new(window_size),
            out: RingBuffer::new(padded_window_size).initialize(0.0),
            window_size,
            ir,
            ir_fft_cache,
            fft_planner,
        }
    }
    pub fn window_size(&self) -> usize {
        self.window_size
    }
    pub fn output_buffer(&self) -> &RingBuffer<f64> {
        &self.out
    }
    pub fn internal_buffer_size(&self) -> usize {
        self.out.len()
    }
    fn padded_window_size(ir_size: usize, window_size: usize) -> usize {
        (ir_size + window_size - 1).next_power_of_two()
    }
}
impl Filter for FFTConvolution {
    fn clear(&mut self) {
        self.x.clear();
        self.out.initialize_again(0.0);
    }
    fn compute(&mut self, signal: f64) -> f64 {
        let buffered_signal = self.out.pop_front().unwrap();
        self.out.push_back(0.0);

        if let Some(chunk) = self.x.buffer_back(Complex::new(signal, 0.0)) {
            let window_size = chunk.len();
            let padded_window_size = Self::padded_window_size(self.ir.len(), window_size);
            let mut buffer: Vec<Complex<f64>> = chunk.into_iter().chain(iter::repeat(Complex::zero()).take(padded_window_size - window_size)).collect();
            {
                let fft = self.fft_planner.lock().unwrap().plan_fft_forward(padded_window_size);
                fft.process(&mut buffer);
            }
            for (i, val) in buffer.iter_mut().enumerate() {
                *val *= self.ir_fft_cache[i];
            }
            {
                let ifft = self.fft_planner.lock().unwrap().plan_fft_inverse(padded_window_size);
                ifft.process(&mut buffer);
            }
            for (out_ref, buf_val) in self.out.inner_mut().iter_mut().zip(buffer.into_iter()).take(padded_window_size) {
                *out_ref += buf_val.re / padded_window_size as f64; //TODO: Magnitude or Real part?
            }
        }
        
        buffered_signal
    }
}


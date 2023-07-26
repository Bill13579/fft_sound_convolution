use std::ops::RangeBounds;

#[cfg(not(feature="slice-ring-buffer"))]
use std::collections::VecDeque;
#[cfg(not(feature="slice-ring-buffer"))]
type BaseDequeImplementation<T> = VecDeque<T>;

#[cfg(feature="slice-ring-buffer")]
use slice_ring_buffer::SliceRingBuffer;
#[cfg(feature="slice-ring-buffer")]
type BaseDequeImplementation<T> = SliceRingBuffer<T>;

#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    inner: BaseDequeImplementation<T>,
    capacity: usize,
    length: usize
}

impl<T> From<Vec<T>> for RingBuffer<T> where T: Clone {
    fn from(value: Vec<T>) -> Self {
        Self::from(BaseDequeImplementation::from_iter(value.into_iter()))
    }
}
impl<T> From<BaseDequeImplementation<T>> for RingBuffer<T> where T: Clone {
    fn from(value: BaseDequeImplementation<T>) -> Self {
        Self::from_deque(value)
    }
}

impl<T> RingBuffer<T> where T: Clone {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: BaseDequeImplementation::with_capacity(capacity),
            capacity,
            length: 0
        }
    }
    pub fn from_deque(deque: BaseDequeImplementation<T>) -> RingBuffer<T> {
        Self {
            capacity: deque.len(),
            length: deque.len(),
            inner: deque
        }
    }
    pub fn len(&self) -> usize {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn into_deque(self) -> BaseDequeImplementation<T> {
        self.inner
    }
    #[cfg(not(feature="slice-ring-buffer"))]
    pub fn inner(&self) -> &BaseDequeImplementation<T> {
        &self.inner
    }
    #[cfg(feature="slice-ring-buffer")]
    pub fn inner(&self) -> &[T] {
        &self.inner
    }
    #[cfg(not(feature="slice-ring-buffer"))]
    /// Note
    /// ====
    /// While this function exists in order to allow for complex operations that
    ///  are not handled by the included utility functions, its use **can** cause
    ///  the `RingBuffer` to become out of sync with its inner deque.
    /// **To avoid that, do not modify the length of the inner deque directly without going
    ///  through one of the utility functions!**
    pub fn inner_mut(&mut self) -> &mut BaseDequeImplementation<T> {
        &mut self.inner
    }
    #[cfg(feature="slice-ring-buffer")]
    pub fn inner_mut(&mut self) -> &mut [T] {
        &mut self.inner
    }
    pub fn clear(&mut self) {
        self.inner.clear();
        self.length = 0;
    }
    pub fn drain<R>(&mut self, range: R) -> Vec<T>
    where
        R: RangeBounds<usize> {
        let drain: Vec<T> = self.inner.drain(range).collect();
        self.length -= drain.len();
        drain
    }
    pub fn empty(&mut self) -> Vec<T> {
        self.length = 0;
        self.inner.drain(..).collect()
    }
    pub fn initialize_again(&mut self, value: T) {
        for _ in 0..self.capacity {
            self.push_back(value.clone());
        }
    }
    pub fn initialize(mut self, value: T) -> Self {
        self.initialize_again(value);
        self
    }
    pub fn back(&self) -> Option<&T> {
        if self.capacity == 0 { return None; }
        self.inner.back()
    }
    pub fn back_mut(&mut self) -> Option<&mut T> {
        if self.capacity == 0 { return None; }
        self.inner.back_mut()
    }
    pub fn back_n(&self, n: usize) -> Option<&T> {
        if self.capacity == 0 || n >= self.inner.len() { return None; }
        self.inner.get(self.inner.len()-n-1)
    }
    pub fn front(&self) -> Option<&T> {
        if self.capacity == 0 { return None; }
        self.inner.front()
    }
    pub fn front_mut(&mut self) -> Option<&mut T> {
        if self.capacity == 0 { return None; }
        self.inner.front_mut()
    }
    pub fn front_n(&self, n: usize) -> Option<&T> {
        if self.capacity == 0 || n >= self.inner.len() { return None; }
        self.inner.get(n)
    }
    pub fn pop_front(&mut self) -> Option<T> {
        if self.capacity == 0 { return None; }
        let item = self.inner.pop_front();
        if let Some(_) = item {
            self.length -= 1;
        }
        item
    }
    pub fn pop_back(&mut self) -> Option<T> {
        if self.capacity == 0 { return None; }
        let item = self.inner.pop_back();
        if let Some(_) = item {
            self.length -= 1;
        }
        item
    }
    pub fn fill_front(&mut self, item: T) {
        while self.length < self.capacity {
            self.push_front(item.clone());
        }
    }
    pub fn fill_back(&mut self, item: T) {
        while self.length < self.capacity {
            self.push_back(item.clone());
        }
    }
    pub fn to_capacity_front(&mut self, capacity: Option<usize>) {
        if let Some(capacity) = capacity {
            self.capacity = capacity;
        }
        while self.length > self.capacity {
            self.pop_front();
        }
    }
    pub fn to_capacity_back(&mut self, capacity: Option<usize>) {
        if let Some(capacity) = capacity {
            self.capacity = capacity;
        }
        while self.length > self.capacity {
            self.pop_back();
        }
    }
    pub fn push_back(&mut self, item: T) {
        if self.capacity == 0 { return; }

        // while self.length >= self.capacity {
        //     self.pop_front();
        // }
        self.to_capacity_front(None);
        if self.length == self.capacity {
            self.pop_front();
        }
        
        self.inner.push_back(item);
        self.length += 1;
    }
    pub fn push_front(&mut self, item: T) {
        if self.capacity == 0 { return; }

        // while self.length >= self.capacity {
        //     self.pop_back();
        // }
        self.to_capacity_back(None);
        if self.length == self.capacity {
            self.pop_back();
        }

        self.inner.push_front(item);
        self.length += 1;
    }
}

pub trait ChunkedBuffer<T> where T: Clone {
    fn buffer_back(&mut self, item: T) -> Option<BaseDequeImplementation<T>>;
    fn buffer_front(&mut self, item: T) -> Option<BaseDequeImplementation<T>>;
}

impl<T> ChunkedBuffer<T> for RingBuffer<T> where T: Clone {
    fn buffer_back(&mut self, item: T) -> Option<BaseDequeImplementation<T>> {
        self.push_back(item);
        if self.length == self.capacity {
            let buf = self.inner.clone();
            self.clear();
            Some(buf)
        } else {
            None
        }
    }
    fn buffer_front(&mut self, item: T) -> Option<BaseDequeImplementation<T>> {
        self.push_front(item);
        if self.length == self.capacity {
            let buf = self.inner.clone();
            self.clear();
            Some(buf)
        } else {
            None
        }
    }
}


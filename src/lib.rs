//! Slab allocator that preserves memory location of elements even after resize. Thus it can be
//! used for [pinning](https://doc.rust-lang.org/std/pin/index.html).
//!
//! Kernel friendly, since it returns an error on allocation failure instead of panicking.

#![no_std]
#![feature(allocator_api)]
#![deny(missing_docs)]
#![deny(clippy::missing_safety_doc)]

use core::fmt::{Debug, Formatter, Result};
use core::mem::size_of;
use core::num::NonZeroU32;
use core::ptr::NonNull;

use alloc::boxed::Box;

const NUM_SEGMENTS: usize = 26;
const SKIP_SEGMENTS: usize = 6;
const _SA: () = _static_assert();

const fn _static_assert() {
    assert!(size_of::<Option<Key>>() == size_of::<Key>());
}

extern crate alloc;

#[cfg(test)]
#[macro_use]
extern crate std;

#[derive(Debug)]
enum Entry<T> {
    Free(Option<Key>),
    Occupied(T),
}

/// Slab allocator for uniform data type
pub struct Slab<T> {
    storage: [Option<NonNull<Entry<T>>>; NUM_SEGMENTS],
    len: usize,
    segments: u32,
    next: Option<Key>,
}

impl<T> Slab<T> {
    /// Creates new allocator
    pub fn new() -> Self {
        Self {
            storage: [const { None }; NUM_SEGMENTS],
            len: 0,
            segments: 0,
            next: None,
        }
    }

    /// Allocated capacity
    pub fn capacity(&self) -> usize {
        ((1 << SKIP_SEGMENTS) << self.segments) - (1 << SKIP_SEGMENTS)
    }

    /// Adds new element to the slab
    pub fn add(&mut self, val: T) -> Option<Key> {
        if let Some(next) = self.next.take() {
            let entry = self.key_to_entry(&next);
            let mut new_entry = Entry::Occupied(val);

            core::mem::swap(&mut new_entry, entry);

            let old_entry = new_entry;

            match old_entry {
                Entry::Free(key) => self.next = key,
                _ => unreachable!(),
            }

            self.len += 1;
            Some(next)
        } else {
            self.storage[self.segments as usize] = Some(Self::new_segment(self.segments)?);
            self.segments += 1;

            self.next = Some(Key::new(self.segments - 1, 0));
            self.add(val)
        }
    }

    /// Retrieves element by key. `key` must be allocated by `add` method of the same slab
    /// instance
    pub fn get(&self, key: &Key) -> Option<&T> {
        // SAFETY: key can be constructed only by this crate, so we expect it to be always valid.
        // If user passes key that is not associated with this slab it's UB
        unsafe {
            let seg = self.storage.get(key.segment())?;
            let entry = (*seg)?.add(key.index());

            match entry.as_ref() {
                Entry::Occupied(val) => Some(val),
                _ => None,
            }
        }
    }

    /// Removes element by key and returns previously inserted value
    pub fn remove(&mut self, key: Key) -> T {
        let mut new_entry = if let Some(next) = self.next.take() {
            Entry::Free(Some(next))
        } else {
            Entry::Free(None)
        };

        let entry = self.key_to_entry(&key);
        core::mem::swap(&mut new_entry, entry);

        match new_entry {
            Entry::Occupied(t) => t,
            _ => unreachable!(),
        }
    }

    fn segment_len(idx: u32) -> u32 {
        (1 << SKIP_SEGMENTS) << idx
    }

    fn new_segment(idx: u32) -> Option<NonNull<Entry<T>>> {
        let len = Self::segment_len(idx);
        let mut new = Box::try_new_uninit_slice(len as usize).ok()?;

        for (i, entry) in new[..len as usize - 1].iter_mut().enumerate() {
            let i = i as u32;

            entry.write(Entry::Free(Some(Key::new(idx, i + 1))));
        }

        new[len as usize - 1].write(Entry::Free(None));

        unsafe {
            let new = new.assume_init();
            Some(NonNull::new_unchecked(Box::leak(new).as_mut_ptr()))
        }
    }

    fn key_to_entry(&mut self, key: &Key) -> &mut Entry<T> {
        // SAFETY: key can be constructed only by this crate, so we expect it to be always valid
        // If user passes key that is not associated with this slab it's UB.
        unsafe {
            self.storage[key.segment()]
                .unwrap_unchecked()
                .add(key.index())
                .as_mut()
        }
    }
}

impl<T: core::fmt::Debug> Default for Slab<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Key that allows to access data
#[derive(PartialEq)]
pub struct Key(NonZeroU32);

impl Key {
    const SEGMENT_BITS: u32 = (NUM_SEGMENTS + 1).next_power_of_two().ilog2();
    const SEGMENT_MASK: u32 = (1 << Self::SEGMENT_BITS) - 1;

    fn new(seg: u32, index: u32) -> Self {
        debug_assert!(seg < 1 << Self::SEGMENT_BITS);

        // SAFETY: since use seg + 1, this operation won't give 0
        unsafe { Self(NonZeroU32::new_unchecked(seg + 1) | index << Self::SEGMENT_BITS) }
    }

    fn index(&self) -> usize {
        (self.0.get() >> Self::SEGMENT_BITS) as usize
    }

    fn segment(&self) -> usize {
        (self.0.get() & Self::SEGMENT_MASK) as usize - 1
    }
}

impl Debug for Key {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "Key {{ segment: {}, index: {} }}",
            self.segment(),
            self.index()
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn basic() {
        let mut slab = Slab::new();

        for i in 0..1 << SKIP_SEGMENTS {
            assert_eq!(slab.add(i), Some(Key::new(0, i)));
            assert_eq!(slab.get(&Key::new(0, i)), Some(&i));
        }

        for i in 0..2 << SKIP_SEGMENTS {
            assert_eq!(slab.add(i), Some(Key::new(1, i)));
            assert_eq!(slab.get(&Key::new(1, i)), Some(&i));
        }

        assert_eq!(slab.add(0), Some(Key::new(2, 0)));
    }

    #[test]
    fn remove() {
        let mut slab = Slab::new();
        let bound = 2_000_000;

        let keys = (1..=bound).map(|x| slab.add(x).unwrap()).collect::<Vec<_>>();

        let sum: u64 = keys.iter().map(|x| slab.get(x).unwrap()).sum();
        assert_eq!(sum, (bound * (bound + 1)) / 2);

        let sum: u64 = keys.into_iter().map(|x| slab.remove(x)).sum();
        assert_eq!(sum, (bound * (bound + 1)) / 2);
    }

    #[test]
    fn fuzz() {
        
    }

    // TODO: more tests =)
}

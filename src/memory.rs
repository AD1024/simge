use crate::sim;
use std::{collections::{BTreeMap, HashSet}};

use egg::Id;

#[derive(Debug)]
pub struct SRAM {
    pub residence: BTreeMap<Id, usize>,
    pub evict: HashSet<Id>,
    pub resident_size: usize,
    pub mem_limit: usize,
    pub trip_count: usize,
}

pub struct DRAM {
    pub residence: BTreeMap<Id, usize>,
}

impl DRAM {
    pub fn new() -> Self {
        Self {
            residence: BTreeMap::new(),
        }
    }
}

impl sim::Memory<Id, DRAM> for SRAM {
    fn put(&mut self, id: &Id, size: usize, from_self: bool) -> bool {
        if size + self.resident_size <= self.mem_limit {
            assert!(!self.residence.contains_key(id));
            self.resident_size += size;
            if !from_self {
                self.trip_count += 1;
            }
            self.residence.insert(id.clone(), size);
            true
        } else {
            panic!("OOM on SRAM: trying to allocate {}; usage: {} / {}", size, self.size_allocated(), self.size_total());
        }
    }

    fn to_vec(&self) -> Vec<&Id> {
        self.residence.iter().map(|pi| pi.0).collect()
    }

    fn size_available(&self) -> usize {
        self.mem_limit - self.resident_size
    }

    fn size_allocated(&self) -> usize {
        self.resident_size
    }

    fn size_total(&self) -> usize {
        self.mem_limit
    }

    fn size_of(&self, data: &Id) -> Result<usize, ()> {
        if let Some(x) = self.residence.get(data) {
            Ok(x.clone())
        } else {
            Err(())
        }
    }

    fn get(&self, id: &Id) -> usize {
        if let Some(size) = self.residence.get(id) {
            size.clone()
        } else {
            panic!("no residence has id {} in SRAM", id);
        }
    }

    fn store(&mut self, id: &Id, evict: bool, dram: &mut DRAM) {
        if self.residence.contains_key(id) {
            let size = self.residence.get(id).unwrap().clone();
            if evict {
                self.residence.remove(id).unwrap();
                self.evict.insert(id.clone());
                self.resident_size -= size;
            }
            self.trip_count += 1;
            dram.put(id, size, false);
        } else {
            panic!("Evicting non-residence: {}", id);
        }
    }

    fn reset(&mut self) {
        self.residence.clear();
        self.evict.clear();
        self.resident_size = 0;
    }

    fn deallocate(&mut self, data: &Id) {
        assert!(self.residence.contains_key(data));
        self.resident_size -= self.residence.get(data).unwrap();
        self.residence.remove(data);
    }

    fn contains(&self, data: &Id) -> bool {
        self.residence.contains_key(data)
    }
}

impl sim::Memory<Id, DRAM> for DRAM {
    fn to_vec(&self) -> Vec<&Id> {
        self.residence.iter().map(|pi| pi.0).collect()
    }

    fn put(&mut self, data: &Id, size: usize, _from_self: bool) -> bool {
        self.residence.insert(data.clone(), size);
        return true;
    }

    fn get(&self, data: &Id) -> usize {
        if let Some(size) = self.residence.get(data) {
            size.clone()
        } else {
            panic!("No resident has id {} in DRAM", data)
        }
    }

    fn size_of(&self, data: &Id) -> Result<usize, ()> {
        if let Some(x) = self.residence.get(data) {
            Ok(x.clone())
        } else {
            Err(())
        }
    }

    fn size_allocated(&self) -> usize {
        0
    }

    fn size_total(&self) -> usize {
        usize::MAX
    }

    fn size_available(&self) -> usize {
        // assumption: Host memory has no limit
        usize::MAX
    }

    fn store(&mut self, _: &Id, _: bool, _: &mut DRAM) {
        return;
    }

    fn reset(&mut self) {
        self.residence.clear();
    }

    fn deallocate(&mut self, data: &Id) {
        self.residence.remove(data);
    }

    fn contains(&self, data: &Id) -> bool {
        self.residence.contains_key(data)
    }
}

impl SRAM {
    pub fn new(sram_size: usize) -> Self {
        Self {
            residence: BTreeMap::default(),
            evict: HashSet::default(),
            resident_size: 0,
            mem_limit: sram_size,
            trip_count: 0,
        }
    }
}

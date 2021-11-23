use crate::sim;
use std::collections::{BTreeMap, HashSet};

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

impl sim::Memory<Id, DRAM> for SRAM {
    fn put(&mut self, id: &Id, size: usize) -> bool {
        if size + self.resident_size < self.mem_limit {
            self.residence.insert(id.clone(), size);
            true
        } else {
            false
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
            dram.put(id, size);
        } else {
            panic!("Evicting non-residence: {}", id);
        }
    }

    fn reset(&mut self) {
        self.residence.clear();
        self.evict.clear();
    }

    fn contains(&self, data: &Id) -> bool {
        self.residence.contains_key(data)
    }
}

impl sim::Memory<Id, DRAM> for DRAM {
    fn put(&mut self, data: &Id, size: usize) -> bool {
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

    fn store(&mut self, _: &Id, _: bool, _: &mut DRAM) {
        return;
    }

    fn reset(&mut self) {
        self.residence.clear();
    }

    fn contains(&self, data: &Id) -> bool {
        self.residence.contains_key(data)
    }
}

impl SRAM {
    fn new(sram_size: usize) -> Self {
        Self {
            residence: BTreeMap::default(),
            evict: HashSet::default(),
            resident_size: 0,
            mem_limit: sram_size,
            trip_count: 0,
        }
    }
}

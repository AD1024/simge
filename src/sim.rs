use std::{
    borrow::BorrowMut,
    collections::{HashMap, HashSet},
};

use egg::Id;

use crate::memory::{self, DRAM, SRAM};

pub trait Simulator<I, D, TM, HM>
where
    I: Instruction<D, TM, HM>,
    HM: Memory<D, HM>,
    TM: Memory<D, HM>,
{
    fn initialize(&mut self);
    fn run_insn(&mut self, insns: I) -> usize;
}

pub trait DTR<I, D, TM, HM>
where
    I: Instruction<D, TM, HM>,
    HM: Memory<D, HM>,
    TM: Memory<D, HM>,
{
    fn rematerialize(&mut self, data: &D, sram: &mut TM, dram: &mut HM, exclude: &HashSet<Id>);
    fn perform_op(&mut self, op: &I, srams: &mut HashMap<String, SRAM>, dram: &mut HM, exclude: &HashSet<Id>);
    fn allocate_buffer(&mut self, size: usize, mem: &mut TM, dram: &mut HM, exclude: &HashSet<Id>);
    fn evict_single(&mut self, exclude: &HashSet<Id>, mem: &mut TM, dram: &mut HM);
    fn deallocate(&mut self, data: &D, mem: &mut TM, dram: &mut HM);
}

pub trait Memory<D, HM>
where
    HM: Memory<D, HM>,
{
    fn put(&mut self, data: &D, size: usize) -> bool;
    fn get(&self, data: &D) -> usize;
    fn contains(&self, data: &D) -> bool;
    fn store(&mut self, data: &D, evict: bool, other: &mut HM) {
        other.put(data, self.get(data));
    }
    fn reset(&mut self);
}

#[derive(Eq, PartialEq, Debug)]
pub enum InsnType {
    Compute,
    MMIO,
}

#[derive(Clone, Debug)]
pub enum Operators {
    /// Execute a sequence of computes
    Compute(String, Id, Id, Vec<(Id, Operators)>, usize),
    /// (Load region data)
    /// Loading data from host to device
    Load(String, (Id, Box<Operators>), usize),
    /// (Store region evict data)
    /// Storing result back to device
    /// If the second field is set to true, on-device memory will be evicted
    Store(String, bool, (Id, Box<Operators>), usize),
    NoOp,
}

pub struct JitSim;

impl JitSim {
    pub fn run(&mut self, ops: &mut Operators, srams: &mut HashMap<String, SRAM>, dram: &mut DRAM, pin: &HashSet<Id>) {
        match ops {
            Operators::NoOp => {},
            Operators::Load(region, meta_data, size) => {
                self.run(meta_data.1.borrow_mut(), srams, dram, pin);
                if *region == String::from("host") {
                    dram.put(&meta_data.0, size.clone());
                } else {
                    self.perform_op(ops, srams, dram, &HashSet::default());
                }
            }
            Operators::Store(region, evict, meta_data, _size) => {
                self.run(meta_data.1.borrow_mut(), srams, dram, pin);
                if *region == String::from("host") {
                    return;
                } else {
                    self.perform_op(ops, srams, dram, &HashSet::default());
                }
            }
            Operators::Compute(_region, _op, _dst, subops, _size) => {
                let pin = subops.iter().map(|x| x.0).collect::<HashSet<_>>();
                for op in subops.iter_mut() {
                    self.run(&mut op.1, srams, dram, &HashSet::default());
                }
                self.perform_op(ops, srams, dram, &pin);
            }
        }
    }
}

impl DTR<Operators, Id, SRAM, DRAM> for JitSim {
    fn rematerialize(
        &mut self,
        data: &Id,
        sram: &mut SRAM,
        dram: &mut DRAM,
        evict_exclude: &HashSet<Id>,
    ) {
        if sram.contains(data) {
            return;
        } else {
            let data_size = dram.get(data);
            self.allocate_buffer(data_size, sram, dram, evict_exclude);
            sram.put(data, data_size.clone());
        }
    }

    fn perform_op(
        &mut self,
        op: &Operators,
        srams: &mut HashMap<String, SRAM>,
        dram: &mut DRAM,
        exclude: &HashSet<Id>,
    ) {
        match op {
            Operators::Compute(region, _, _, ids, size) => {
                if *region == String::from("host") {
                    op.run(None, dram);
                } else {
                    let mem = srams.get_mut(region).unwrap();
                    let evict_lock = ids.iter().cloned().map(|x| x.0).collect::<HashSet<_>>();
                    for arg in ids.iter().map(|x| x.0) {
                        if !mem.contains(&arg) {
                            self.rematerialize(&arg, mem, dram, &evict_lock);
                        }
                    }
                    self.allocate_buffer(size.clone(), mem, dram, &evict_lock);
                    op.run(Some(mem), dram);
                }
            }
            Operators::Load(region, (_id, _op), size) => {
                if *region == String::from("host") {
                    op.run(None, dram);
                } else {
                    let mem = srams.get_mut(region).unwrap();
                    self.allocate_buffer(size.clone(), mem, dram, exclude);
                    op.run(Some(mem), dram);
                }
            }
            Operators::Store(region, _evict, (_data, _op), _size) => {
                if *region == String::from("host") {
                    op.run(None, dram);
                } else {
                    let mem = srams.get_mut(region).unwrap();
                    op.run(Some(mem), dram);
                }
            }
            Operators::NoOp => {}
        }
    }

    fn allocate_buffer(
        &mut self,
        size: usize,
        mem: &mut SRAM,
        dram: &mut DRAM,
        exclude: &HashSet<Id>,
    ) {
        while mem.resident_size + size > mem.mem_limit {
            self.evict_single(exclude, mem, dram);
        }
    }

    fn evict_single(&mut self, exclude: &HashSet<Id>, mem: &mut SRAM, dram: &mut DRAM) {
        let mut evict = None;
        for (id, _) in mem.residence.iter() {
            if !exclude.contains(id) {
                evict = Some(id.clone());
            }
        }
        if let Some(ev) = evict {
            mem.store(&ev, true, dram);
        } else {
            panic!("Thrashes here...")
        }
    }

    fn deallocate(&mut self, data: &Id, mem: &mut SRAM, dram: &mut DRAM) {
        assert!(mem.contains(data));
        mem.store(data, true, dram);
    }
}

impl Instruction<Id, memory::SRAM, memory::DRAM> for Operators {
    fn insn_type(&self) -> InsnType {
        match self {
            &Operators::Compute(_, _, _, _, _) => InsnType::Compute,
            _ => InsnType::MMIO,
        }
    }

    fn run(&self, mem: Option<&mut memory::SRAM>, dram: &mut memory::DRAM) {
        match self {
            Self::Compute(region, _, output_id, ids, size) => {
                let _op = ids[0].0;
                // TODO: could do interpreter here but not necessary
                // we are only generating schedule a la DTR
                if *region == String::from("host") {
                    assert!(ids[1..].iter().all(|x| dram.contains(&x.0)));
                    dram.put(output_id, size.clone());
                } else {
                    if let Some(mem) = mem {
                        assert!(ids[1..].iter().all(|x| mem.contains(&x.0)));
                        assert!(mem.mem_limit > mem.resident_size + size);
                        mem.put(output_id, size.clone());
                    } else {
                        panic!("No SRAM provided");
                    }
                }
            }
            Self::Load(region, (data, _op), size) => {
                if *region == String::from("host") {
                    dram.put(data, size.clone());
                } else {
                    assert!(dram.contains(data));
                    assert!(mem.is_some());
                    let mem = mem.unwrap();
                    assert!(mem.mem_limit > mem.resident_size + size);
                    mem.trip_count += 1;
                    mem.put(data, size.clone());
                }
            }
            Self::Store(_, evict, (data, _op), _) => {
                assert!(mem.is_some());
                let mem = mem.unwrap();
                assert!(mem.contains(data));
                mem.store(data, *evict, dram);
            }
            Self::NoOp => {}
        }
    }

    fn compile(&self) -> String {
        match &self {
            Operators::Compute(region, op, output_id, ids, _) => format!(
                "(compute {} {} {} {})",
                region,
                op,
                output_id,
                ids.iter()
                    .map(|x| format!("{}", x.0))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            Operators::Load(region, (data, _op), _) => format!("(load {} {})", region, data),
            Operators::Store(region, evict, (data, _op), _) => {
                format!("(store {} {} {})", region, evict, data)
            }
            Operators::NoOp => "Skip".into(),
        }
    }
}

impl InsnLogger<Id, SRAM, DRAM> for Operators {
    fn write_log(self, logs: &mut Vec<String>) {
        logs.push(self.compile());
    }
}

pub trait Instruction<D, TM, HM>
where
    TM: Memory<D, HM>,
    HM: Memory<D, HM>,
{
    fn insn_type(&self) -> InsnType;
    fn run(&self, mem: Option<&mut memory::SRAM>, dram: &mut memory::DRAM);
    fn compile(&self) -> String;
}

pub trait InsnLogger<D, TM, HM>
where
    TM: Memory<D, HM>,
    HM: Memory<D, HM>,
{
    fn write_log(self, logs: &mut Vec<String>);
}

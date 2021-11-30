use std::{
    borrow::BorrowMut,
    collections::{HashMap, HashSet},
    hash::Hash,
    marker::PhantomData,
};

use egg::Id;
use log::info;

use crate::memory::{DRAM, SRAM};

pub trait Simulator<I, D, TM, HM>
where
    D: std::fmt::Debug + Hash + Eq + PartialEq + Clone,
    I: Instruction<D>,
    HM: Memory<D, HM>,
    TM: Memory<D, HM>,
{
    fn initialize(&mut self);
    fn run_insn(&mut self, insns: I) -> usize;
}

pub trait DTR<I, D, TM, HM>
where
    D: std::fmt::Debug + Hash + Eq + PartialEq + Clone,
    I: Instruction<D>,
    HM: Memory<D, HM>,
    TM: Memory<D, HM>,
{
    fn rematerialize(&mut self, data: &D, sram: &mut TM, dram: &mut HM, exclude: &HashSet<D>);
    fn perform_op(
        &mut self,
        op: &I,
        srams: &mut HashMap<String, TM>,
        dram: &mut HM,
        exclude: &HashSet<D>,
    );
    fn allocate_buffer(&mut self, size: usize, mem: &mut TM, dram: &mut HM, exclude: &HashSet<D>);
    fn evict_single(&mut self, exclude: &HashSet<D>, mem: &mut TM, dram: &mut HM);
    fn deallocate(&mut self, data: &D, mem: &mut TM, dram: &mut HM);
}

pub trait Heuristic<D, TM, HM>
where
    D: std::fmt::Debug + Hash + Eq + PartialEq + Clone,
    TM: Memory<D, HM>,
    HM: Memory<D, HM>,
{
    fn choose(&mut self, sram: &TM, exclude: &HashSet<D>) -> Option<D>;
    fn touch(&mut self, data: &D);
}

pub trait Memory<D, HM>
where
    D: std::fmt::Debug + Hash + Eq + PartialEq + Clone,
    HM: Memory<D, HM>,
{
    fn put(&mut self, data: &D, size: usize) -> bool;
    fn get(&self, data: &D) -> usize;
    fn contains(&self, data: &D) -> bool;
    fn size_available(&self) -> usize;
    fn size_allocated(&self) -> usize;
    fn size_total(&self) -> usize;
    fn to_vec(&self) -> Vec<&D>;
    fn store(&mut self, data: &D, _evict: bool, other: &mut HM) {
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
pub enum Operators<D>
where
    D: std::fmt::Debug,
{
    /// Execute a sequence of computes
    Compute(String, D, D, Vec<(D, Operators<D>)>, usize),
    /// (Load region data)
    /// Loading data from host to device
    Load(String, (D, Box<Operators<D>>), usize),
    /// (Store region evict data)
    /// Storing result back to device
    /// If the second field is set to true, on-device memory will be evicted
    Store(String, bool, (D, Box<Operators<D>>), usize),
    NoOp,
}

pub struct JitSim<H, D, TM, HM>
where
    D: std::fmt::Debug + Hash + Eq + PartialEq + Clone,
    TM: Memory<D, HM>,
    HM: Memory<D, HM>,
    H: Heuristic<D, TM, HM>,
{
    pub heuristic: H,
    __phantom_d: PhantomData<D>,
    __phantom_tm: PhantomData<TM>,
    __phantom_hm: PhantomData<HM>,
}

impl<H, D, TM, HM> JitSim<H, D, TM, HM>
where
    D: std::fmt::Debug + Hash + Eq + PartialEq + Clone,
    TM: Memory<D, HM>,
    HM: Memory<D, HM>,
    H: Heuristic<D, TM, HM>,
{
    pub fn new(heuristic: H) -> Self {
        Self {
            heuristic,
            __phantom_d: PhantomData,
            __phantom_hm: PhantomData,
            __phantom_tm: PhantomData,
        }
    }

    pub fn run(
        &mut self,
        ops: &mut Operators<D>,
        srams: &mut HashMap<String, TM>,
        dram: &mut HM,
        pin: &HashSet<D>,
    ) {
        match ops {
            Operators::NoOp => {}
            Operators::Load(_region, meta_data, _size) => {
                self.run(meta_data.1.borrow_mut(), srams, dram, pin);
                self.perform_op(ops, srams, dram, &HashSet::default());
            }
            Operators::Store(_region, _evict, meta_data, _size) => {
                self.run(meta_data.1.borrow_mut(), srams, dram, pin);
                self.perform_op(ops, srams, dram, &HashSet::default());
            }
            Operators::Compute(_region, _op, _dst, subops, _size) => {
                let pin = subops.iter().map(|x| &x.0).cloned().collect::<HashSet<_>>();
                for op in subops.iter_mut() {
                    self.run(&mut op.1, srams, dram, &pin);
                }
                self.perform_op(ops, srams, dram, &HashSet::default());
            }
        }
    }
}

impl<H, D, TM, HM> DTR<Operators<D>, D, TM, HM> for JitSim<H, D, TM, HM>
where
    D: std::fmt::Debug + Hash + Eq + PartialEq + Clone,
    TM: Memory<D, HM>,
    HM: Memory<D, HM>,
    H: Heuristic<D, TM, HM>,
{
    fn rematerialize(
        &mut self,
        data: &D,
        sram: &mut TM,
        dram: &mut HM,
        evict_exclude: &HashSet<D>,
    ) {
        self.heuristic.touch(data);
        if sram.contains(data) {
            return;
        } else {
            info!("Rematerialize {:?}", data);
            let data_size = dram.get(data);
            self.allocate_buffer(data_size, sram, dram, evict_exclude);
            sram.put(data, data_size.clone());
        }
    }

    fn perform_op(
        &mut self,
        op: &Operators<D>,
        srams: &mut HashMap<String, TM>,
        dram: &mut HM,
        exclude: &HashSet<D>,
    ) {
        match op {
            Operators::Compute(region, _, _, ids, size) => {
                if *region == String::from("host") {
                    op.run(None as Option<&mut TM>, dram);
                } else {
                    let mem = srams.get_mut(region).unwrap();
                    let evict_lock = ids.iter().map(|x| &x.0).cloned().collect::<HashSet<_>>();
                    for arg in ids.iter().map(|x| x.0.clone()) {
                        if !mem.contains(&arg) {
                            self.rematerialize(&arg, mem, dram, &evict_lock);
                        } else {
                            self.heuristic.touch(&arg);
                        }
                    }
                    self.allocate_buffer(size.clone(), mem, dram, &evict_lock);
                    op.run(Some(mem), dram);
                }
            }
            Operators::Load(region, (id, _op), size) => {
                if *region == String::from("host") {
                    op.run(None as Option<&mut TM>, dram);
                } else {
                    self.heuristic.touch(id);
                    let mem = srams.get_mut(region).unwrap();
                    self.allocate_buffer(size.clone(), mem, dram, exclude);
                    op.run(Some(mem), dram);
                }
            }
            Operators::Store(region, _evict, (_data, _op), _size) => {
                if *region == String::from("host") {
                    panic!("Store should not performed on host");
                } else {
                    let mem = srams.get_mut(region).unwrap();
                    op.run(Some(mem), dram);
                }
            }
            Operators::NoOp => {}
        }
    }

    fn allocate_buffer(&mut self, size: usize, mem: &mut TM, dram: &mut HM, exclude: &HashSet<D>) {
        while mem.size_allocated() + size >= mem.size_total() {
            self.evict_single(exclude, mem, dram);
        }
    }

    fn evict_single(&mut self, exclude: &HashSet<D>, mem: &mut TM, dram: &mut HM) {
        // for id in mem.residence.iter().map(|x| x.0).filter(|&x| !exclude.contains(x)) {
        //     if !exclude.contains(id) {
        //         evict = Some(id.clone());
        //     }
        // }
        // let allowed = mem.residence.iter().map(|x| x.0).filter(|&&x| !exclude.contains(&x)).cloned().collect::<Vec<_>>();
        // if allowed.len() > 0 {
        //     let x = allowed.choose(&mut rand::thread_rng());
        //     if let Some(x) = x {
        //         evict = Some(x);
        //     }
        // }
        if let Some(ev) = self.heuristic.choose(mem, exclude) {
            info!("Evict: {:?}", ev);
            mem.store(&ev, true, dram);
        } else {
            panic!("Thrashes here...")
        }
    }

    fn deallocate(&mut self, data: &D, mem: &mut TM, dram: &mut HM) {
        assert!(mem.contains(data));
        mem.store(data, true, dram);
    }
}

impl<D> Instruction<D> for Operators<D>
where
    D: std::fmt::Debug + Hash + Eq + PartialEq + Clone,
{
    fn insn_type(&self) -> InsnType {
        match self {
            &Operators::Compute(_, _, _, _, _) => InsnType::Compute,
            _ => InsnType::MMIO,
        }
    }

    fn run<TM: Memory<D, HM>, HM: Memory<D, HM>>(&self, mem: Option<&mut TM>, dram: &mut HM) {
        match self {
            Self::Compute(region, _, output_id, ids, size) => {
                let _op = &ids[0].0;
                info!(
                    "Current Op: Compute {} {:?}",
                    region,
                    ids.iter().map(|x| &x.0).cloned().collect::<Vec<_>>()
                );
                // TODO: could do interpreter here but not necessary
                // we are only generating schedule a la DTR
                if *region == String::from("host") {
                    assert!(ids.iter().all(|x| dram.contains(&x.0)));
                    dram.put(output_id, size.clone());
                } else {
                    if let Some(mem) = mem {
                        assert!(ids.iter().all(|x| mem.contains(&x.0)));
                        assert!(mem.size_total() > mem.size_allocated() + size);
                        mem.put(output_id, size.clone());
                    } else {
                        panic!("No SRAM provided");
                    }
                }
            }
            Self::Load(region, (data, _op), size) => {
                info!("Current Op: Load {} {:?}", region, data);
                if *region == String::from("host") {
                    dram.put(data, size.clone());
                } else {
                    assert!(dram.contains(data));
                    assert!(mem.is_some());
                    let mem = mem.unwrap();
                    assert!(mem.size_total() > mem.size_allocated() + size);
                    mem.put(data, size.clone());
                }
            }
            Self::Store(region, evict, (data, _op), _) => {
                info!("Current Op: Store {} {:?} evict: {}", region, data, evict);
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
                "(compute {} {:?} {:?} {})",
                region,
                op,
                output_id,
                ids.iter()
                    .map(|x| format!("{:?}", x.0))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            Operators::Load(region, (data, _op), _) => format!("(load {} {:?})", region, data),
            Operators::Store(region, evict, (data, _op), _) => {
                format!("(store {} {} {:?})", region, evict, data)
            }
            Operators::NoOp => "Skip".into(),
        }
    }
}

impl InsnLogger<Id, SRAM, DRAM> for Operators<Id> {
    fn write_log(self, logs: &mut Vec<String>) {
        logs.push(self.compile());
    }
}

pub trait Instruction<D>
where
    D: std::fmt::Debug + Hash + Eq + PartialEq + Clone,
{
    fn insn_type(&self) -> InsnType;
    fn run<TM: Memory<D, HM>, HM: Memory<D, HM>>(&self, mem: Option<&mut TM>, dram: &mut HM);
    fn compile(&self) -> String;
}

pub trait InsnLogger<D, TM, HM>
where
    D: std::fmt::Debug + Hash + Eq + PartialEq + Clone,
    TM: Memory<D, HM>,
    HM: Memory<D, HM>,
{
    fn write_log(self, logs: &mut Vec<String>);
}

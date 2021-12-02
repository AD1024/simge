use crate::sim::{Heuristic, Memory};
use rand::seq::SliceRandom;
use std::{collections::BinaryHeap, collections::HashSet, hash::Hash, time::Instant};

pub struct RandomEviction;

impl RandomEviction {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Clone, Debug)]
struct DataPair<D: Clone>(Instant, D);

impl<D: Clone> PartialEq for DataPair<D> {
    fn eq(&self, other: &Self) -> bool {
        self.0.elapsed() == other.0.elapsed()
    }
}

impl<D: Clone> PartialOrd for DataPair<D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0
            .elapsed()
            .as_nanos()
            .partial_cmp(&other.0.elapsed().as_nanos())
    }
}

impl<D: Clone> Eq for DataPair<D> {}

impl<D: Clone> Ord for DataPair<D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.0.elapsed().as_nanos() < other.0.elapsed().as_nanos() {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    }
}
pub struct LRU<D: Clone> {
    member: BinaryHeap<DataPair<D>>,
}

impl<D> Heuristic<D> for RandomEviction
where
    D: std::fmt::Debug + Hash + Eq + PartialEq + Clone,
{
    fn choose<TM>(&mut self, sram: &TM, exclude: &HashSet<D>) -> Option<D>
    where
        TM: Memory<D>,
    {
        let allowed = sram
            .to_vec()
            .iter()
            .filter(|&&x| !exclude.contains(&x))
            .cloned()
            .collect::<Vec<_>>();
        if allowed.len() > 0 {
            let x = allowed.choose(&mut rand::thread_rng());
            if let Some(&x) = x {
                return Some(x.clone());
            }
        }
        return None;
    }

    fn touch(&mut self, _data: &D, _size: usize) {}
    fn evict(&mut self, _data: &D) {}
}

impl<D: Clone> LRU<D> {
    pub fn new() -> Self {
        LRU {
            member: BinaryHeap::default(),
        }
    }
}

impl<D> Heuristic<D> for LRU<D>
where
    D: std::fmt::Debug + Hash + Eq + PartialEq + Clone,
{
    fn choose<TM>(&mut self, _sram: &TM, exclude: &HashSet<D>) -> Option<D>
    where
        TM: Memory<D>,
    {
        let x = self
            .member
            .iter()
            .filter(|&x| !exclude.contains(&x.1))
            .collect::<BinaryHeap<_>>();
        let x = x.peek();
        x.map_or(None, |&x| Some(x.1.clone()))
    }

    fn touch(&mut self, data: &D, _size: usize) {
        self.evict(data);
        self.member.push(DataPair(Instant::now(), data.clone()));
    }

    fn evict(&mut self, data: &D) {
        self.member = self
            .member
            .iter()
            .filter(|&x| x.1 != data.clone())
            .cloned()
            .collect::<BinaryHeap<_>>();
    }
}

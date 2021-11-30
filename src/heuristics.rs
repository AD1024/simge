use crate::sim::{Heuristic, Memory};
use rand::seq::SliceRandom;
use std::{collections::HashSet, hash::Hash};

pub struct RandomEviction;

impl<D, TM, HM> Heuristic<D, TM, HM> for RandomEviction
where
    D: std::fmt::Debug + Hash + Eq + PartialEq + Clone,
    TM: Memory<D, HM>,
    HM: Memory<D, HM>,
{
    fn choose(&mut self, sram: &TM, exclude: &HashSet<D>) -> Option<D> {
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

    fn touch(&mut self, _data: &D) {}
}

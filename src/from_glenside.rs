use std::collections::HashSet;

use egg::{EGraph, Id, RecExpr};
use glenside::language::{Language, MyAnalysis, MyAnalysisData};
use ndarray::Dimension;

use crate::sim::Operators;

pub fn compile_instruction(
    current_id: &Id,
    expr: &RecExpr<Language>,
    memo: &mut HashSet<Id>,
    egraph: &EGraph<Language, MyAnalysis>,
) -> Operators {
    if memo.contains(current_id) {
        return Operators::NoOp;
    }
    memo.insert(current_id.clone());
    let node = expr.nodes[usize::from(current_id.clone())].clone();
    let mut insn = vec![];
    match node {
        Language::RelayOperatorCall(ids) => {
            assert!(ids.len() > 1);
            if let Language::RelayOperator(_) = expr.nodes[usize::from(ids[0])] {
                for children_id in ids[1..].iter() {
                    insn.push(compile_instruction(children_id, expr, memo, egraph));
                }
                let output_size = match &egraph[current_id.clone()].data {
                    MyAnalysisData::AccessPattern(access) => access.as_vec().iter().product(),
                    MyAnalysisData::Shape(shape) => shape.shape.slice().to_vec().iter().product(),
                    _ => panic!(),
                };
                return Operators::Compute(
                    "host".into(),
                    ids[0],
                    current_id.clone(),
                    ids[1..].iter().cloned().zip(insn.into_iter()).collect(),
                    output_size,
                );
            } else {
                panic!(
                    "Expecting a RelayOperator, got {:?}",
                    expr.nodes[usize::from(ids[0])]
                );
            }
        }
        Language::AcceleratorLoad([region, data]) => {
            let load_cmd = compile_instruction(&data, expr, memo, egraph);
            let region = match &egraph[region].data {
                MyAnalysisData::AcceleratorFunc(func) => func.accelerator.clone().into(),
                _ => panic!("Not a valid accelerator store"),
            };
            let output_size = match &egraph[data].data {
                MyAnalysisData::AccessPattern(access) => access.as_vec().iter().product(),
                MyAnalysisData::Shape(shape) => shape.shape.slice().to_vec().iter().product(),
                _ => panic!(),
            };
            // (accelerator-call <region> <loads..>)
            // accelerator calls will use the ids of their direct children
            // therefore we store the id of `Load` here.
            return Operators::Load(
                region,
                (current_id.clone(), Box::new(load_cmd)),
                output_size,
            );
        }
        Language::AcceleratorStore([region, data]) => {
            let store_cmd = compile_instruction(&data, expr, memo, egraph);
            let region = match &egraph[region].data {
                MyAnalysisData::AcceleratorFunc(func) => func.accelerator.clone().into(),
                _ => panic!("Not a valid accelerator store"),
            };
            let output_size = match &egraph[data].data {
                MyAnalysisData::AccessPattern(access) => access.as_vec().iter().product(),
                MyAnalysisData::Shape(shape) => shape.shape.slice().to_vec().iter().product(),
                _ => panic!(),
            };
            // Store could be used by multiple parents
            // According to the rewrite rule, a store will be merged with a parent
            // load if and only if the load is the only parent to the store
            // therefore an overestimation is if there are multiple parents of store,
            // we keep it on-device.
            if egraph[current_id.clone()].parents.len() > 1 {
                return Operators::Store(
                    region,
                    false,
                    (data.clone(), Box::new(store_cmd)),
                    output_size,
                );
            } else {
                return Operators::Store(
                    region,
                    true,
                    (data.clone(), Box::new(store_cmd)),
                    output_size,
                );
            }
        }
        Language::AcceleratorCall(ids) => {
            let ids = ids.to_vec();
            for children_id in ids[1..].iter() {
                insn.push(compile_instruction(children_id, expr, memo, egraph));
            }
            let region = match &egraph[ids[0]].data {
                MyAnalysisData::AcceleratorFunc(func) => func.accelerator.clone().into(),
                _ => panic!("Not a valid accelerator store"),
            };
            let output_size = match &egraph[current_id.clone()].data {
                MyAnalysisData::AccessPattern(access) => access.as_vec().iter().product(),
                MyAnalysisData::Shape(shape) => shape.shape.slice().to_vec().iter().product(),
                _ => panic!(),
            };
            return Operators::Compute(
                region,
                ids[0],
                current_id.clone(),
                ids[1..].iter().cloned().zip(insn.into_iter()).collect(),
                output_size,
            );
        }
        Language::AccessTensor(access) => {
            let output_size = match &egraph[current_id.clone()].data {
                MyAnalysisData::AccessPattern(access) => access.as_vec().iter().product(),
                MyAnalysisData::Shape(shape) => shape.shape.slice().to_vec().iter().product(),
                _ => panic!(),
            };
            return Operators::Load(
                "host".into(),
                (access, Box::new(Operators::NoOp)),
                output_size,
            );
        }
        _ => panic!("Not supported: {:?}", node),
    }
}

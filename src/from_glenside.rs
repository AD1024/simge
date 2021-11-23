use std::{borrow::Borrow, collections::{HashMap, HashSet}};

use egg::{EGraph, Id, RecExpr};
use glenside::language::{Language, MyAnalysis, MyAnalysisData};
use ndarray::Dimension;

use crate::sim::Operators;

pub fn compile_instruction(
    current_id: &Id,
    expr: &RecExpr<Language>,
    memo: &mut HashMap<Id, Id>,
    egraph: &EGraph<Language, MyAnalysis>,
    id_translation: &HashMap<Id, Id>,
) -> (Operators, Id) {
    let current_id = id_translation.get(current_id).unwrap();
    if memo.contains_key(current_id) {
        return (Operators::NoOp, memo.get(current_id).unwrap().clone());
    }
    let node = expr.nodes[usize::from(current_id.clone())].clone();
    println!("Current node: {:?} {}", node, current_id);
    let mut insn = vec![];
    match node {
        Language::RelayOperatorCall(ids) => {
            assert!(ids.len() > 1);
            if let Language::RelayOperator(_) = expr.nodes[usize::from(ids[0])] {
                let mut mem_id = vec![];
                for children_id in ids[1..].iter() {
                    let (op, id) = compile_instruction(children_id, expr, memo, egraph, id_translation);
                    insn.push(op);
                    mem_id.push(id);
                }
                let ids = ids.to_vec().iter().cloned().map(|x| id_translation.get(&x).unwrap().clone()).collect::<Vec<_>>();
                // let output_size = match &egraph[current_id.clone()].data {
                //     MyAnalysisData::AccessPattern(access) => access.as_vec().iter().product(),
                //     MyAnalysisData::Shape(shape) => shape.shape.slice().to_vec().iter().product(),
                //     _ => panic!(),
                // };
                memo.insert(current_id.clone(), current_id.clone());
                return (Operators::Compute(
                    "host".into(),
                    ids[0],
                    current_id.clone(),
                    mem_id.iter().cloned().zip(insn.into_iter()).collect(),
                    1,
                ), current_id.clone());
            } else {
                panic!(
                    "Expecting a RelayOperator, got {:?}",
                    expr.nodes[usize::from(ids[0])]
                );
            }
        }
        Language::AcceleratorLoad([region, data]) => {
            let (load_cmd, src_id) = compile_instruction(&data, expr, memo, egraph, id_translation);
            let region = id_translation.get(&region).unwrap().clone();
            let region = match &egraph[region].data {
                MyAnalysisData::AcceleratorFunc(func) => func.accelerator.clone().into(),
                _ => panic!("Not a valid accelerator load: {:?}", egraph[region].data),
            };
            // let output_size = match &egraph[data].data {
            //     MyAnalysisData::AccessPattern(access) => access.as_vec().iter().product(),
            //     MyAnalysisData::Shape(shape) => shape.shape.slice().to_vec().iter().product(),
            //     _ => panic!(),
            // };
            // (accelerator-call <region> <loads..>)
            // accelerator calls will use the ids of their direct children
            // therefore we store the id of `Load` here.
            memo.insert(current_id.clone(), src_id.clone());
            return (Operators::Load(
                region,
                (src_id.clone(), Box::new(load_cmd)),
                1,
            ), src_id.into());
        }
        Language::AcceleratorStore([region, data]) => {
            let (store_cmd, dst_id) = compile_instruction(&data, expr, memo, egraph, id_translation);
            let region = id_translation.get(&region).unwrap().clone();
            let region = match &egraph[region].data {
                MyAnalysisData::AcceleratorFunc(func) => func.accelerator.clone().into(),
                _ => panic!("Not a valid accelerator store: {:?}", egraph[region].data),
            };
            // let output_size = match &egraph[data].data {
            //     MyAnalysisData::AccessPattern(access) => access.as_vec().iter().product(),
            //     MyAnalysisData::Shape(shape) => shape.shape.slice().to_vec().iter().product(),
            //     _ => panic!(),
            // };
            // Store could be used by multiple parents
            // According to the rewrite rule, a store will be merged with a parent
            // load if and only if the load is the only parent to the store
            // therefore an overestimation is if there are multiple parents of store,
            // we keep it on-device.
            // if egraph[current_id.clone()].parents.len() > 1 {
            //     return (Operators::Store(
            //         region,
            //         false,
            //         (dst_id.clone(), Box::new(store_cmd)),
            //         1,
            //     ), dst_id.clone());
            // } else {
                memo.insert(current_id.clone(), dst_id.clone());
                return (Operators::Store(
                    region,
                    true,
                    (dst_id.clone(), Box::new(store_cmd)),
                    1,
                ), dst_id.into());
            // }
        }
        Language::AcceleratorCall(ids) => {
            let mut mem_id = vec![];
            for children_id in ids[1..ids.len() - 1].iter() {
                let (op, id) = compile_instruction(children_id, expr, memo, egraph, id_translation);
                insn.push(op);
                mem_id.push(id);
            }
            let ids = ids.to_vec().iter().map(|x| id_translation.get(x).unwrap().clone()).collect::<Vec<_>>();
            let region = match &egraph[ids[0]].data {
                MyAnalysisData::AcceleratorFunc(func) => func.accelerator.clone().into(),
                _ => panic!("Not a valid accelerator store"),
            };
            // let output_size = match &egraph[current_id.clone()].data {
            //     MyAnalysisData::AccessPattern(access) => access.as_vec().iter().product(),
            //     MyAnalysisData::Shape(shape) => shape.shape.slice().to_vec().iter().product(),
            //     _ => panic!(),
            // };
            memo.insert(current_id.clone(), current_id.clone());
            return (Operators::Compute(
                region,
                ids[0],
                current_id.clone(),
                mem_id.iter().cloned().zip(insn.into_iter()).collect(),
                1,
            ), current_id.clone());
        }
        Language::AccessTensor(_) => {
            // let output_size = match &egraph[current_id.clone()].data {
            //     MyAnalysisData::AccessPattern(access) => access.as_vec().iter().product(),
            //     MyAnalysisData::Shape(shape) => shape.shape.slice().to_vec().iter().product(),
            //     _ => panic!(),
            // };
            memo.insert(current_id.clone(), current_id.clone());
            return (Operators::Load(
                "host".into(),
                (current_id.clone(), Box::new(Operators::NoOp)),
                1,
            ), current_id.clone());
        }
        _ => panic!("Not supported: {:?}", node),
    }
}

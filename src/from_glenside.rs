use std::collections::HashMap;

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
) -> Option<(Operators<Id>, Id)> {
    let current_id = id_translation.get(current_id).unwrap();
    if memo.contains_key(current_id) {
        return Some((Operators::NoOp, memo.get(current_id).unwrap().clone()));
    }
    let node = expr.nodes[usize::from(current_id.clone())].clone();
    let mut insn = vec![];
    match node {
        Language::RelayOperatorCall(ids) => {
            assert!(ids.len() > 1);
            if let Language::RelayOperator(_) = expr.nodes[usize::from(ids[0])] {
                let mut mem_id = vec![];
                for children_id in ids[1..].iter() {
                    if let Some((op, id)) =
                        compile_instruction(children_id, expr, memo, egraph, id_translation) {
                        insn.push(op);
                        mem_id.push(id);
                    }
                }
                assert!(insn.len() > 0, "Empty children at accelerator call {:?}", egraph[ids[0]].nodes);
                let ids = ids
                    .to_vec()
                    .iter()
                    .cloned()
                    .map(|x| id_translation.get(&x).unwrap().clone())
                    .collect::<Vec<_>>();
                // let output_size = match &egraph[current_id.clone()].data {
                //     MyAnalysisData::AccessPattern(access) => access.as_vec().iter().product(),
                //     MyAnalysisData::Shape(shape) => shape.shape.slice().to_vec().iter().product(),
                //     _ => panic!(),
                // };
                memo.insert(current_id.clone(), current_id.clone());
                return Some((
                    Operators::Compute(
                        "host".into(),
                        ids[0],
                        current_id.clone(),
                        mem_id.iter().cloned().zip(insn.into_iter()).collect(),
                        1,
                    ),
                    current_id.clone(),
                ));
            } else {
                panic!(
                    "Expecting a RelayOperator, got {:?}",
                    expr.nodes[usize::from(ids[0])]
                );
            }
        }
        Language::AcceleratorLoad([region, data]) => {
            let (load_cmd, src_id) = compile_instruction(&data, expr, memo, egraph, id_translation).unwrap();
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
            return Some((
                Operators::Load(region, (src_id.clone(), Box::new(load_cmd)), 1),
                src_id.into(),
            ));
        }
        Language::AcceleratorStore([region, data]) => {
            let (store_cmd, dst_id) =
                compile_instruction(&data, expr, memo, egraph, id_translation).unwrap();
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
            return Some((
                Operators::Store(region, false, (dst_id.clone(), Box::new(store_cmd)), 1),
                dst_id.into(),
            ));
            // }
        }
        Language::AcceleratorCall(ids) => {
            let mut mem_id = vec![];
            for children_id in ids[1..ids.len() - 1].iter() {
                if let Some((op, id)) = compile_instruction(children_id, expr, memo, egraph, id_translation) {
                    insn.push(op);
                    mem_id.push(id);
                }
            }
            assert!(insn.len() > 0, "Empty children at accelerator call {:?}", egraph[ids[0]].nodes);
            let ids = ids
                .to_vec()
                .iter()
                .map(|x| id_translation.get(x).unwrap().clone())
                .collect::<Vec<_>>();
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
            return Some((
                Operators::Compute(
                    region,
                    ids[0],
                    current_id.clone(),
                    mem_id.iter().cloned().zip(insn.into_iter()).collect(),
                    1,
                ),
                current_id.clone(),
            ));
        }
        Language::Compute([op, x]) => {
            let (child_op, id) = compile_instruction(&x, expr, memo, egraph, id_translation).unwrap();
            memo.insert(current_id.clone(), current_id.clone());
            return Some((
                Operators::Compute(
                    "host".into(), op, current_id.clone(), vec![(id, child_op)], 1
                ),
                current_id.clone(),
            ))
        }
        Language::AccessPair([car, cdr]) => {
            let mut child_insn = vec![];
            if let Some((car_op, car_id)) = compile_instruction(&car, expr, memo, egraph, id_translation) {
                child_insn.push((car_id, car_op));
            }
            if let Some((cdr_op, cdr_id)) = compile_instruction(&cdr, expr, memo, egraph, id_translation) {
                child_insn.push((cdr_id, cdr_op));
            }
            memo.insert(current_id.clone(), current_id.clone());
            if child_insn.len() > 0 {
                return Some((
                    Operators::Compute(
                        "host".into(), current_id.clone(), current_id.clone(), child_insn, 1
                    ),
                    current_id.clone(),
                ));
            } else {
                return None;
            }
        }
        Language::AccessInsertAxis([x, _])
        | Language::AccessBroadcast([x, _])
        | Language::Access([x, _]) => {
            return compile_instruction(&x, expr, memo, egraph, id_translation);
        }
        Language::AccessLiteral(_)
        | Language::AccessTensor(_) => {
            // let output_size = match &egraph[current_id.clone()].data {
            //     MyAnalysisData::AccessPattern(access) => access.as_vec().iter().product(),
            //     MyAnalysisData::Shape(shape) => shape.shape.slice().to_vec().iter().product(),
            //     _ => panic!(),
            // };
            memo.insert(current_id.clone(), current_id.clone());
            return Some((
                Operators::Load(
                    "host".into(),
                    (current_id.clone(), Box::new(Operators::NoOp)),
                    1,
                ),
                current_id.clone(),
            ));
        }
        Language::AccessFlatten(x) => {
            let op = compile_instruction(&x, expr, memo, egraph, id_translation).unwrap();
            return Some((Operators::Compute("host".into(), current_id.clone(), current_id.clone(), vec![(op.1, op.0)], 1), current_id.clone()));
        }
        Language::RelayActivationLayout(_)
        | Language::Usize(_)
        | Language::Shape(_)
        | Language::RelayKernelLayout(_) => None,
        _ => panic!("Not supported: {:?}", node),
    }
}

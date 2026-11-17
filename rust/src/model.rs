//! ONNX models on the `rten` runtime, loaded from a file or from memory.
//!
//! `beat-this`'s [`RtenRuntime`](beat_this::RtenRuntime) loads models from
//! paths only; in the browser (wasm32) there is no file system, so the models
//! arrive as bytes. [`OnnxModel`] implements the crate's [`Model`] trait over
//! either source, with the same name mapping of inputs and outputs.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use beat_this::{Model, Tensor};
use rten::{Model as RtenGraph, NodeId, ValueView};
use rten_tensor::Layout;

/// Where the models of one analysis come from.
#[derive(Debug, Clone, Copy)]
pub enum ModelSource<'a> {
    /// A directory with the model files (desktop).
    Dir(&'a Path),
    /// The model files' bytes (browser). `drum` and `chord` are `None` when
    /// that model was not provided.
    Bytes {
        mel: &'a [u8],
        beat: &'a [u8],
        drum: Option<&'a [u8]>,
        chord: Option<&'a [u8]>,
    },
}

/// A loaded ONNX model addressed by input/output names.
pub struct OnnxModel {
    model: RtenGraph,
    inputs: HashMap<String, NodeId>,
    outputs: Vec<(NodeId, String)>,
}

impl OnnxModel {
    /// Loads the model file at `path` (memory-mapped by `rten` where possible).
    pub fn from_file(path: &Path) -> Result<Self> {
        Ok(Self::wrap(RtenGraph::load_file(path)?))
    }

    /// Loads a model from its file's bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self::wrap(RtenGraph::load(bytes.to_vec())?))
    }

    fn wrap(model: RtenGraph) -> Self {
        let name = |id: NodeId| {
            model
                .node_info(id)
                .and_then(|info| info.name().map(str::to_string))
        };
        let inputs = model
            .input_ids()
            .iter()
            .filter_map(|&id| Some((name(id)?, id)))
            .collect();
        let outputs = model
            .output_ids()
            .iter()
            .map(|&id| (id, name(id).unwrap_or_else(|| format!("output_{id:?}"))))
            .collect();
        Self {
            model,
            inputs,
            outputs,
        }
    }
}

impl Model for OnnxModel {
    fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<HashMap<String, Tensor>> {
        let values = inputs
            .iter()
            .map(|(name, tensor)| {
                let id = *self
                    .inputs
                    .get(*name)
                    .ok_or_else(|| anyhow!("unknown model input '{name}'"))?;
                let view = ValueView::from_shape(tensor.shape.as_slice(), tensor.data.as_slice())
                    .map_err(|e| anyhow!("input tensor '{name}': {e}"))?;
                Ok((id, view.into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let ids: Vec<NodeId> = self.outputs.iter().map(|(id, _)| *id).collect();
        let values = self.model.run(values, &ids, None)?;
        let mut result = HashMap::with_capacity(values.len());
        for ((_, name), value) in self.outputs.iter().zip(values) {
            let tensor = value
                .into_tensor::<f32>()
                .ok_or_else(|| anyhow!("model output '{name}' is not f32"))?;
            let shape = tensor.shape().to_vec();
            result.insert(
                name.clone(),
                Tensor {
                    shape,
                    data: tensor.into_data(),
                },
            );
        }
        Ok(result)
    }
}

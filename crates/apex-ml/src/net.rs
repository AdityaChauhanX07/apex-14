//! 1D circular convolutional network for raceline prediction.
//!
//! Predicts speed and lateral offset profiles from track geometry features.
//! Uses circular padding to handle the wrap-around nature of closed tracks.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder, VarMap};

/// Apply circular (wrap-around) padding to a 1D tensor.
///
/// For a tensor of shape [batch, channels, length], pads by copying
/// the last `pad` elements to the front and the first `pad` elements
/// to the back, producing shape [batch, channels, length + 2*pad].
fn circular_pad_1d(x: &Tensor, pad: usize) -> Result<Tensor> {
    let len = x.dim(2)?;
    let suffix = x.narrow(2, len - pad, pad)?;
    let prefix = x.narrow(2, 0, pad)?;
    Tensor::cat(&[&suffix, x, &prefix], 2)
}

/// 1D circular CNN for raceline prediction.
///
/// Architecture:
///   Conv1D(4->32, k=7, circular) + ReLU
///   Conv1D(32->64, k=5, circular) + ReLU
///   Conv1D(64->64, k=5, circular) + ReLU
///   Conv1D(64->32, k=3, circular) + ReLU
///   Conv1D(32->2, k=1) -- pointwise output
///
/// Input: [batch, 4, N_FIXED] (curvature, curvature_deriv, width_left, width_right)
/// Output: [batch, 2, N_FIXED] (predicted speed, predicted offset)
pub struct RacelineNet {
    conv1: Conv1d,
    conv2: Conv1d,
    conv3: Conv1d,
    conv4: Conv1d,
    conv_out: Conv1d,
}

impl RacelineNet {
    /// Build the network from a VarBuilder (for training with VarMap).
    pub fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let cfg = Conv1dConfig::default();
        let conv1 = candle_nn::conv1d(4, 32, 7, cfg, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv1d(32, 64, 5, cfg, vb.pp("conv2"))?;
        let conv3 = candle_nn::conv1d(64, 64, 5, cfg, vb.pp("conv3"))?;
        let conv4 = candle_nn::conv1d(64, 32, 3, cfg, vb.pp("conv4"))?;
        let conv_out = candle_nn::conv1d(32, 2, 1, cfg, vb.pp("conv_out"))?;
        Ok(Self {
            conv1,
            conv2,
            conv3,
            conv4,
            conv_out,
        })
    }

    /// Forward pass. Input shape: [batch, 4, N_FIXED]. Output shape: [batch, 2, N_FIXED].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = circular_pad_1d(x, 3)?;
        let x = self.conv1.forward(&x)?.relu()?;

        let x = circular_pad_1d(&x, 2)?;
        let x = self.conv2.forward(&x)?.relu()?;

        let x = circular_pad_1d(&x, 2)?;
        let x = self.conv3.forward(&x)?.relu()?;

        let x = circular_pad_1d(&x, 1)?;
        let x = self.conv4.forward(&x)?.relu()?;

        self.conv_out.forward(&x)
    }

    /// Predict speed and offset profiles from track features.
    ///
    /// Takes the four feature channels (each of length N_FIXED) and returns
    /// predicted (speed, offset) profiles (each of length N_FIXED).
    /// All values are in normalized space (same as training data).
    pub fn predict(
        &self,
        curvature: &[f64],
        curvature_deriv: &[f64],
        width_left: &[f64],
        width_right: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let device = Device::Cpu;
        let n = curvature.len();
        let mut input_data = Vec::with_capacity(4 * n);
        for slice in [curvature, curvature_deriv, width_left, width_right] {
            input_data.extend(slice.iter().map(|&v| v as f32));
        }
        let input = Tensor::from_vec(input_data, (1, 4, n), &device)?;
        let output = self.forward(&input)?;

        let speed_t = output.get(0)?.get(0)?;
        let offset_t = output.get(0)?.get(1)?;

        let speed = speed_t
            .to_vec1::<f32>()?
            .iter()
            .map(|&v| v as f64)
            .collect();
        let offset = offset_t
            .to_vec1::<f32>()?
            .iter()
            .map(|&v| v as f64)
            .collect();
        Ok((speed, offset))
    }

    /// Count the total number of trainable parameters.
    pub fn param_count(&self) -> usize {
        let layers: [&Conv1d; 5] = [
            &self.conv1,
            &self.conv2,
            &self.conv3,
            &self.conv4,
            &self.conv_out,
        ];
        let mut total = 0usize;
        for layer in layers {
            total += layer.weight().elem_count();
            if let Some(bias) = layer.bias() {
                total += bias.elem_count();
            }
        }
        total
    }
}

/// Save trained network weights to a safetensors file.
pub fn save_weights(var_map: &VarMap, path: &std::path::Path) -> Result<()> {
    var_map.save(path)
}

/// Load a trained network from a safetensors file.
///
/// Returns the network ready for inference.
pub fn load_network(path: &std::path::Path) -> Result<RacelineNet> {
    let device = Device::Cpu;
    let mut var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let net = RacelineNet::new(vb)?;
    var_map.load(path)?;
    Ok(net)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    use crate::data::N_FIXED;

    fn make_net() -> RacelineNet {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        RacelineNet::new(vb).expect("network creation failed")
    }

    #[test]
    fn test_network_construction() {
        let _net = make_net();
    }

    #[test]
    fn test_forward_shape() {
        let net = make_net();
        let device = Device::Cpu;
        let input = Tensor::zeros((1, 4, N_FIXED), DType::F32, &device).expect("zeros");
        let output = net.forward(&input).expect("forward");
        assert_eq!(output.dims(), &[1, 2, N_FIXED]);
    }

    #[test]
    fn test_forward_batch() {
        let net = make_net();
        let device = Device::Cpu;
        let input = Tensor::zeros((3, 4, N_FIXED), DType::F32, &device).expect("zeros");
        let output = net.forward(&input).expect("forward");
        assert_eq!(output.dims(), &[3, 2, N_FIXED]);
    }

    #[test]
    fn test_predict() {
        let net = make_net();
        let zeros = vec![0.0f64; N_FIXED];
        let (speed, offset) = net
            .predict(&zeros, &zeros, &zeros, &zeros)
            .expect("predict");
        assert_eq!(speed.len(), N_FIXED);
        assert_eq!(offset.len(), N_FIXED);
    }

    #[test]
    fn test_param_count() {
        let net = make_net();
        let count = net.param_count();
        // Architecture: 4->32 k=7, 32->64 k=5, 64->64 k=5, 64->32 k=3, 32->2 k=1 + biases
        // Actual count is ~38k; prompt estimated ~30k (deviation noted below).
        assert!(
            count > 25_000 && count < 50_000,
            "param count {count} out of expected range"
        );
    }

    #[test]
    fn test_circular_pad_1d() {
        let device = Device::Cpu;
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let t = Tensor::from_vec(data, (1, 1, 5), &device).expect("from_vec");
        let padded = circular_pad_1d(&t, 2).expect("pad");
        assert_eq!(padded.dims(), &[1, 1, 9]);
        let vals: Vec<f32> = padded
            .squeeze(0)
            .expect("sq0")
            .squeeze(0)
            .expect("sq1")
            .to_vec1()
            .expect("to_vec1");
        // last 2 before data, first 2 after: [4,5, 1,2,3,4,5, 1,2]
        assert_eq!(vals, vec![4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0]);
    }

    #[test]
    fn test_output_not_nan() {
        let net = make_net();
        let device = Device::Cpu;
        let input = Tensor::rand(0.0f32, 1.0f32, (2, 4, N_FIXED), &device).expect("rand");
        let output = net.forward(&input).expect("forward");
        let vals: Vec<f32> = output
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("to_vec1");
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "output contains NaN or Inf"
        );
    }
}

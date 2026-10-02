use burn::backend::{Autodiff, NdArray};
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::TensorData;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
gsablok@proton.me
 */

type TrainBackend = Autodiff<NdArray<f32>>;
type InferBackend = NdArray<f32>;

#[derive(Module, Debug)]
pub struct SoftMaxRegression<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> SoftMaxRegression<B> {
    pub fn new(inputsize: usize, num_classes: usize, device: &B::Device) -> Self {
        let linear = LinearConfig::new(inputsize, num_classes).init(device);
        Self { linear }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }

    pub fn predict(&self, x: Tensor<B, 2>) -> Tensor<B, 1, Int> {
        self.forward(x).argmax(1).squeeze::<1>()
    }
}

/// Reads a CSV file where the last column is the integer class label and
/// every preceding column is a numeric gene expression from the RNA-se datasets.
fn read_dataset(pathfile: &str) -> Result<(Vec<Vec<f32>>, Vec<i64>), Box<dyn Error>> {
    let fileopen = File::open(pathfile)?;
    let fileread = BufReader::new(fileopen);
    let mut vecinput: Vec<Vec<f32>> = Vec::new();
    let mut labelinput: Vec<i64> = Vec::new();

    for i in fileread.lines() {
        let line = i?;
        if line.trim().is_empty() {
            continue;
        }
        let linevec = line.split(",").collect::<Vec<_>>();
        vecinput.push(
            linevec[0..linevec.len() - 1]
                .iter()
                .map(|x| x.parse::<f32>().unwrap())
                .collect::<Vec<_>>(),
        );
        labelinput.push(linevec[linevec.len() - 1].parse::<i64>()?);
    }

    Ok((vecinput, labelinput))
}

/// Converts row-major feature rows into a 2D Tensor.
fn features_to_tensor<B: Backend>(rows: &[Vec<f32>], device: &B::Device) -> Tensor<B, 2> {
    let num_rows = rows.len();
    let num_cols = rows[0].len();
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    Tensor::<B, 2>::from_data(TensorData::new(flat, [num_rows, num_cols]), device)
}

fn labels_to_tensor<B: Backend>(labels: &[i64], device: &B::Device) -> Tensor<B, 1, Int> {
    let num_rows = labels.len();
    Tensor::<B, 1, Int>::from_data(TensorData::new(labels.to_vec(), [num_rows]), device)
}

pub fn traindata(
    pathfile: &str,
    num_classes: &str,
    model_path: &str,
) -> Result<String, Box<dyn Error>> {
    // BUG FIX: `device` was never defined.
    let device = <TrainBackend as Backend>::Device::default();

    let (vecinput, labelinput) = read_dataset(pathfile)?;
    let inputsize = vecinput[0].len();

    let x = features_to_tensor::<TrainBackend>(&vecinput, &device);
    let y = labels_to_tensor::<TrainBackend>(&labelinput, &device);

    let mut model =
        SoftMaxRegression::<TrainBackend>::new(inputsize, num_classes.parse::<usize>()?, &device);

    let mut optim = SgdConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let lr = 0.5;

    for epoch in 0..300 {
        let logits = model.forward(x.clone());
        let loss = loss_fn.forward(logits, y.clone());
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
        if epoch % 50 == 0 {
            println!("epoch -> {epoch}: loss = {:.4}", loss.into_scalar());
        }
    }
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(model_path, &recorder)
        .expect("failed to save model");

    Ok("Multiclass training has been finished".to_string())
}

/// Loads a previously trained/saved model and runs prediction on a CSV file.
/// The CSV may or may not include a trailing label column; only the feature
/// columns are used (pass `has_labels = true` if the last column should be
/// stripped off before predicting).
pub fn predict_saved_model(
    model_path: &str,
    pathfile: &str,
    inputsize: usize,
    num_classes: usize,
    has_labels: bool,
) -> Result<Vec<i64>, Box<dyn Error>> {
    let device = <InferBackend as Backend>::Device::default();

    // Rebuild the model architecture, then load the saved weights into it.
    let model = SoftMaxRegression::<InferBackend>::new(inputsize, num_classes, &device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = model
        .load_file(model_path, &recorder, &device)
        .expect("failed to load model");

    let fileopen = File::open(pathfile)?;
    let fileread = BufReader::new(fileopen);
    let mut rows: Vec<Vec<f32>> = Vec::new();

    for i in fileread.lines() {
        let line = i?;
        if line.trim().is_empty() {
            continue;
        }
        let linevec = line.split(",").collect::<Vec<_>>();
        let feature_slice = if has_labels {
            &linevec[0..linevec.len() - 1]
        } else {
            &linevec[..]
        };
        rows.push(
            feature_slice
                .iter()
                .map(|v| v.parse::<f32>().unwrap())
                .collect::<Vec<_>>(),
        );
    }

    let x = features_to_tensor::<InferBackend>(&rows, &device);
    let preds = model.predict(x);

    let preds_data = preds.into_data();
    let preds_vec: Vec<i64> = preds_data.convert::<i64>().into_vec().unwrap();

    Ok(preds_vec)
}

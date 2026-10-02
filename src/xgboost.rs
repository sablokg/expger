use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use xgb::parameters::learning::Metrics;
use xgb::{Booster, DMatrix, parameters};

/*
Gaurav Sablok
gsablok@proton.me
*/

/// Reads a CSV file of expression values, builds an XGBoost multiclass
/// classifier (control vs. treatment based on the control mean), trains it,
/// and returns the per-sample class probabilities predicted on the training
/// data.
pub fn expression_xgboost(pathfile: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let mut control1: Vec<f32> = Vec::new();
    let mut control2: Vec<f32> = Vec::new();
    let mut control3: Vec<f32> = Vec::new();

    let mut replicate1_1: Vec<f32> = Vec::new();
    let mut replicate1_2: Vec<f32> = Vec::new();
    let mut replicate1_3: Vec<f32> = Vec::new();

    let mut replicate2_1: Vec<f32> = Vec::new();
    let mut replicate2_2: Vec<f32> = Vec::new();
    let mut replicate2_3: Vec<f32> = Vec::new();

    let mut replicate3_1: Vec<f32> = Vec::new();
    let mut replicate3_2: Vec<f32> = Vec::new();
    let mut replicate3_3: Vec<f32> = Vec::new();

    let fileopen = File::open(pathfile)?;
    let fileread = BufReader::new(fileopen);

    for i in fileread.lines() {
        let line = i?;
        let linesplit = line
            .split(',')
            .map(|x| x.trim().parse::<f32>())
            .collect::<Result<Vec<_>, _>>()?;

        // 12 columns expected: 3 controls + 3 replicate groups x 3 reps each
        control1.push(linesplit[0]);
        control2.push(linesplit[1]);
        control3.push(linesplit[2]);
        replicate1_1.push(linesplit[3]);
        replicate1_2.push(linesplit[4]);
        replicate1_3.push(linesplit[5]);
        replicate2_1.push(linesplit[6]);
        replicate2_2.push(linesplit[7]);
        replicate2_3.push(linesplit[8]);
        replicate3_1.push(linesplit[9]);
        replicate3_2.push(linesplit[10]);
        replicate3_3.push(linesplit[11]);
    }

    /*
    Sum expression values across replicates/controls, then classify samples
    against the control mean. Since this is a classification problem we
    feed the summed counts straight into XGBoost's multiclass softprob
    objective rather than any pre-scaled/sigmoid value.
    */

    let control_add: Vec<f32> = control1
        .iter()
        .zip(control2.iter())
        .zip(control3.iter())
        .map(|((a, b), c)| a + b + c)
        .collect();

    let replicate_1_add: Vec<f32> = replicate1_1
        .iter()
        .zip(replicate1_2.iter())
        .zip(replicate1_3.iter())
        .map(|((a, b), c)| a + b + c)
        .collect();

    let replicate_2_add: Vec<f32> = replicate2_1
        .iter()
        .zip(replicate2_2.iter())
        .zip(replicate2_3.iter())
        .map(|((a, b), c)| a + b + c)
        .collect();

    let replicate_3_add: Vec<f32> = replicate3_1
        .iter()
        .zip(replicate3_2.iter())
        .zip(replicate3_3.iter())
        .map(|((a, b), c)| a + b + c)
        .collect();

    let mean = |v: &[f32]| -> f32 { v.iter().sum::<f32>() / v.len() as f32 };
    let control_mean = mean(&control_add);

    // Feature rows: [control_sum, rep1_sum, rep2_sum, rep3_sum]
    let expression_tuple: Vec<(f32, f32, f32, f32)> = control_add
        .iter()
        .zip(replicate_1_add.iter())
        .zip(replicate_2_add.iter())
        .zip(replicate_3_add.iter())
        .map(|(((c, r1), r2), r3)| (*c, *r1, *r2, *r3))
        .collect();

    // Label: 0 = at/below the control mean, 1 = above it.
    let labels: Vec<i32> = expression_tuple
        .iter()
        .map(|x| if x.0 <= control_mean { 0 } else { 1 })
        .collect();

    let num_class = 2u32;

    let mut expressionmap: Vec<f32> = Vec::with_capacity(expression_tuple.len() * 4);
    for (c, r1, r2, r3) in expression_tuple.iter() {
        expressionmap.push(*c);
        expressionmap.push(*r1);
        expressionmap.push(*r2);
        expressionmap.push(*r3);
    }

    let labelmap: Vec<f32> = labels.iter().map(|x| *x as f32).collect();

    let mut dtrain = DMatrix::from_dense(&expressionmap, expression_tuple.len())?;
    dtrain.set_labels(&labelmap)?;

    let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(parameters::learning::Objective::MultiSoftprob(num_class))
        .eval_metrics(Metrics::Custom(vec![
            parameters::learning::EvaluationMetric::MultiClassLogLoss,
            parameters::learning::EvaluationMetric::MultiClassErrorRate,
        ]))
        .build()?;

    let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
        .max_depth(4)
        .eta(0.3)
        .subsample(0.8)
        .colsample_bytree(0.8)
        .build()?;

    let booster_params = parameters::BoosterParametersBuilder::default()
        .booster_type(parameters::BoosterType::Tree(tree_params))
        .learning_params(learning_params)
        .verbose(true)
        .build()?;

    let training_params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dtrain)
        .boost_rounds(50)
        .booster_params(booster_params)
        .build()?;

    let booster = Booster::train(&training_params)?;

    // Predict on the training matrix. With the MultiSoftprob objective
    // XGBoost already returns a flat vector of per-class probabilities
    // (num_samples * num_class). I reshape that into one row per sample
    // and, defensively, re-normalize with softmax so the result is always
    // a valid probability distribution even if raw margins come back.
    let raw_predictions = booster.predict(&dtrain)?;
    let num_class = num_class as usize;

    let probabilities: Vec<Vec<f32>> = raw_predictions.chunks(num_class).map(softmax).collect();

    Ok(probabilities)
}

/// Converts a row of raw class scores into a probability distribution
/// (values in [0, 1] summing to 1).
fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

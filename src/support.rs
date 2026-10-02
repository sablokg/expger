use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::mean_absolute_error;
use smartcore::model_selection::train_test_split;
use smartcore::svm::Kernels;
use smartcore::svm::svr::SVR;
use smartcore::svm::svr::SVRParameters;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
gsablok@proton.me
 */

/// given your RNA-seq datasets and the fungal pathogenicity value
/// associated with the RNA-seq datasets. The last colum is your
/// pathogenicity score.
pub fn svm(pathfile: &str) -> Result<String, Box<dyn Error>> {
    let fileopen = File::open(pathfile).expect("file not found");
    let fileread = BufReader::new(fileopen);

    let mut inputvec: Vec<Vec<f32>> = Vec::new();
    let mut inputlabel: Vec<f32> = Vec::new();

    for i in fileread.lines() {
        let line = i.expect("line not present");
        let linevec = line.split(",").collect::<Vec<_>>();
        let interline = linevec[0..linevec.len() - 1]
            .iter()
            .map(|x| x.parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        let interlabel = linevec[linevec.len() - 1..linevec.len()]
            .concat()
            .to_string()
            .parse::<f32>()
            .unwrap();
        inputvec.push(interline);
        inputlabel.push(interlabel);
    }

    let densematrix = DenseMatrix::from_2d_vec(&inputvec).unwrap();

    let splitratio = train_test_split(&densematrix, &inputlabel, 0.2, true, Some(49));

    let splitratio_0 = splitratio.0;
    let splitratio_1 = splitratio.1;
    let splitratio_2 = splitratio.2;
    let splitratio_3 = splitratio.3;

    let svmtrain = SVRParameters::default()
        .with_kernel(Kernels::rbf().with_gamma(0.5))
        .with_c(0.1)
        .with_eps(1e-3)
        .with_tol(10.00);

    let svmtrain = SVR::fit(&splitratio_0, &splitratio_2, &svmtrain).unwrap();

    let svmpredict = svmtrain.predict(&splitratio_1).unwrap();

    let error = mean_absolute_error(&splitratio_3, &svmpredict);

    println!("The observed mean absolute error of the model is {}", error);

    Ok("SVM has finished".to_string())
}

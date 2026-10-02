mod args;
use crate::args::CommandParse;
use crate::args::Commands;
use clap::Parser;
use figlet_rs::FIGfont;
mod logisitic;
use crate::logisitic::predict_saved_model;
use crate::logisitic::traindata;
mod batcher;
mod support;
mod xgboost;
use crate::support::svm;

/*
Gaurav Sablok
gsablok@proton.me
*/

#[tokio::main]
async fn main() {
    let fontgenerate = FIGfont::standard().unwrap();
    let repgenerate = fontgenerate.convert("expGER");
    println!("{}", repgenerate.unwrap());

    let args = CommandParse::parse();
    match &args.command {
        Commands::MultiClass {
            pathname,
            numclasses,
            modelpath,
        } => {
            let command = traindata(pathname, numclasses, modelpath).unwrap();

            println!(
                "The command has finished and the file has been written:{}",
                command
            );
        }
        Commands::MultiClassPredict {
            model_path,
            pathfile,
            inputsize,
            num_classes,
            has_labels,
        } => {
            let command =
                predict_saved_model(model_path, pathfile, *inputsize, *num_classes, *has_labels)
                    .unwrap();
            println!("The commands has finished:{:?}", command);
        }
        Commands::SupportVectorMachine { pathfile } => {
            let command = svm(pathfile).unwrap();
            println!("The command has finished:{}", command);
        }
    }
}

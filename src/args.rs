use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "expGER",
    version = "1.0",
    about = "Multiclass Logistic Regression for Expression
       ************************************************
       Gaurav Sablok,
       Email: gsablok@proton.me
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Logistic modelling regression
    MultiClass {
        /// path to the filename
        pathname: String,
        /// number of classes,
        numclasses: String,
        /// model path to be saved
        modelpath: String,
    },
    /// Multi class logistic regression
    MultiClassPredict {
        /// saved model path
        model_path: String,
        /// path to the input data
        pathfile: String,
        /// inputsize means the number of the expression data for each sample
        inputsize: usize,
        /// number of classes
        num_classes: usize,
        /// whether the data is labelled or not
        has_labels: bool,
    },
    /// support vector regression with kernel smoothing
    SupportVectorMachine {
        /// pathfile to the expression datasets
        pathfile: String,
    },
}

extern crate alumine;

use std::str::FromStr;
use std::fs::File;
use std::io::{BufReader,BufRead};

use alumine::alg::Vector;

use alumine::ml::Classifier;
use alumine::ml::linear::LinearRegression;

fn read_data(filename: &str) -> (Vec<Vector<f64>>, Vec<f64>) {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    let mut samples = Vec::new();
    let mut labels = Vec::new();

    for line in reader.lines().filter_map(|line| line.ok()) {
        if line.starts_with("#") { continue; }
        let mut tokens = line.split(",");
        let data = f64::from_str(tokens.next().unwrap()).unwrap();
        let label = f64::from_str(tokens.next().unwrap()).unwrap();
        // Always add a 1 as final value to allow for affine offset
        samples.push(Vector::from_slice(&[data, 1f64]));
        labels.push(label);
    }

    (samples,labels)
}

fn main() {
    let (samples, labels) = read_data("assets/linear.csv");

    let mut classifier = LinearRegression::new();

    classifier.train(&samples, &labels);

    println!("{}", classifier.classify(&Vector::from_slice(&[0f64,1f64])));
    println!("{}", classifier.classify(&Vector::from_slice(&[10f64,1f64])));
    println!("{}", classifier.classify(&Vector::from_slice(&[20f64,1f64])));
}

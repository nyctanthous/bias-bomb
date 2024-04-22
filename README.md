# bias-bomb

Replicates an attack from Ma, Yuzhe and Zhu, Xiaojin and Hsu, Justin's "Data poisoning against differentially-private learners: attacks and defenses" (2019) approach to poison ("bias") training data.
Focuses on label aversion with shallow selection on surrogate victims.

## Running

This project was designed to run Synthea's Covid 19 dataset. At the current time, you can download it at https://synthea.mitre.org/downloads. The 10K entry varient was used for this program.

First, process the Covid-19 data by executing all cells in the `dataset` notebook. Then, switch to the `bias` notebook to perform Ma 

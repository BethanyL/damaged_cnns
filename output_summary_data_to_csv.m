function output_summary_data_to_csv(file_name, accuracies, trial_counter)
% handles saving summary accuracy data to a csv file 

csvwrite(file_name, accuracies);

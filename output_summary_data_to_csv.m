function output_summary_data_to_csv(file_name, accuracies, trial_counter)

csvwrite(file_name, accuracies);

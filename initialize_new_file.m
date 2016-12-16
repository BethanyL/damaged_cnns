function file_name = initialize_new_file(header_string, trial_counter, expnum)
% creates a new .csv file in working directory

file_name = get_file_name(trial_counter, expnum);
fd = fopen(file_name, 'a');
fprintf(fd, header_string);
fclose(fd);

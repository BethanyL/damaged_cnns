% returns new and unique file name
function file_name = get_file_name(trial_counter, expnum)
   file_name = strcat(sprintf('./exp%s/mnist_cnn_exp%s_trial_%d_', expnum, expnum, trial_counter),...
       datestr(now, 'yyyy_mm_dd_HH_MM_SS_FFF'), '.csv');

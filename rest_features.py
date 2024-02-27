import pandas as pd
import numpy as np
import pandas as pd
import pickle
import statistics
import enum
import multiprocessing

clean_tac_path = ""
pickle_path = ""
data_path = ""

class Features(enum.Enum):
	Mean = 0
	Median = 1
	Std_Dev = 2
	Max_Raw = 3
	Min_Raw = 4
	Max_Abs = 5
	Min_Abs = 6


FeatureType = {}
FeatureType[Features.Mean] = 'Mean'
FeatureType[Features.Median] = 'Median'
FeatureType[Features.Std_Dev] = 'Std_Dev'
FeatureType[Features.Max_Raw] = 'Max_Raw'
FeatureType[Features.Min_Raw] = 'Min_Raw'
FeatureType[Features.Max_Abs] = 'Max_Abs'
FeatureType[Features.Min_Abs] = 'Min_Abs'

def process_feature(feature):
    filename = create_per_second_data("BK7610.csv", feature.value)
    return create_per_window_data(filename, feature.value)

def create_per_second_data(pid_filename, metric_no):
	acc_data = pd.read_csv(pid_filename)
	prev_ts = 0
	full_frame = list()
	sub_frame = list()
	tot_rows = len(acc_data)

	for idx in range(0, tot_rows):

		if idx%10000 == 0: print(idx, "**", metric_no)

		r = acc_data.iloc[idx]
		curr_ts = r['time']%1000

		if idx != 0: prev_ts = acc_data.loc[idx-1, 'time']%1000

		if curr_ts > prev_ts:
			sub_frame.append([r['time'], r['x'], r['y'], r['z']])

		else:
			# Do calculations for all enteries of one second
			sub_frame = np.array(sub_frame)
			metrics_axis = []


			# Add the last timstamp in that second window
			metrics_axis.append(sub_frame[-1, 0])

			# Iterating over x, y, z axis
			for col in range(1,4):
				if metric_no == Features.Mean.value:
					metrics_axis.append(sub_frame[:, col].mean())
				elif metric_no == Features.Median.value:
					metrics_axis.append(statistics.median(sub_frame[:, col]))
				elif metric_no == Features.Std_Dev.value:
					metrics_axis.append(statistics.stdev(sub_frame[:, col]))
				elif metric_no == Features.Max_Raw.value:
					metrics_axis.append(max(sub_frame[:, col]))
				elif metric_no == Features.Min_Raw.value:
					metrics_axis.append(min(sub_frame[:, col]))
				elif metric_no == Features.Max_Abs.value:
					metrics_axis.append(max(abs(sub_frame[:, col])))
				elif metric_no == Features.Min_Abs.value:
					metrics_axis.append(min(abs(sub_frame[:, col])))

			full_frame.append(metrics_axis)
			sub_frame = list()

	full_frame = np.array(full_frame)
	
    # Pickling this data
	filename = pickle_path+"Metric_" + str(metric_no) + "_36.pkl"
	outfile = open(filename,'wb')
	pickle.dump(full_frame,outfile)
	outfile.close()

	return filename


def create_per_window_data(filename, metric_no):
	# Read the pickle file that contains entry for each second
	infile = open(filename,'rb')
	mean_all = pickle.load(infile)
	infile.close()

	tot_rows = len(mean_all)
	full_frame = list()
	single_row = list()
	i = 0
	print("Shape of data obatined from pickle - ", mean_all.shape)

	# Calculate summary statistics for this metric
	while i+10 < tot_rows:
		single_row.append(mean_all[i+9:i+10, 0][0])

		for col in range(1, 4):
			sub_frame = mean_all[i:i+10, col]

			single_row.append(sub_frame.mean())
			single_row.append(sub_frame.var())
			single_row.append(sub_frame.max())
			single_row.append(sub_frame.min())
			sub_frame = sorted(sub_frame)
			single_row.append(np.array(sub_frame[0:4]).mean())
			single_row.append(np.array(sub_frame[8:11]).mean())

		full_frame.append(single_row)
		single_row = list()
		i += 10

	full_frame = np.array(full_frame)
	print("Shape of generated frame for each 10 sec window ", full_frame.shape)

	col_names = ['xMe', 'xVr', 'xMx', 'xMi', 'xUM', 'xLM', 'yMe', 'yVr', 'yMx', 'yMn', 'yUM', 'yLM', 'zMe', 'zVr', 'zMx', 'zMi', 'zUM', 'zLM']

	df1 = pd.DataFrame.from_records(full_frame, columns = ['t'] + [str(str(metric_no) + names) for names in col_names] )
	print("df1 created !!!!")

	# Calculating the values out of difference of two windows
	diff_frame = list()
	for i in range(len(full_frame)):
		if i==0: diff_frame.append(full_frame[:1, 1:][0])
		else:
			diff = full_frame[i:i+1, 1:] - full_frame[i-1:i, 1:]
			diff_frame.append(diff[0])

	diff_frame = np.array(diff_frame)
	print("diff_frame created with shape", diff_frame.shape)

	# Generating set2 col name
	df2 = pd.DataFrame.from_records(diff_frame, columns = [str("d" + str(metric_no) + names) for names in col_names])
	print("df2 created !!!!")

	result_df = pd.concat([df1, df2], axis=1)
	outputFileName = pickle_path+"Metric_" + str(metric_no) + "_36.pkl"

	print(result_df.shape)

	# Pickle this data
	outfile = open(outputFileName, 'wb')
	pickle.dump(result_df, outfile)
	outfile.close()
	return outputFileName


def main():
    # Define the pool of processes
    pool = multiprocessing.Pool()

    # Map the process_feature function to each feature using multiprocessing
    filenames = pool.map(process_feature, Features)

    # Close the pool to release resources
    pool.close()
    pool.join()

    return filenames
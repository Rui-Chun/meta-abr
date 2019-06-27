import os


COOKED_TRACE_FOLDER = './mrc_div_train/'


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER, low_tp=0, high_tp=1):
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:

        if cooked_file[0] != str(low_tp) or cooked_file[2] != str(high_tp):
            continue

        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            flag = 0
            for line in f:
                parse = line.split()
                try:
                    float(parse[0])
                except Exception:
                    flag = 1
                    break

                cooked_time.append(float(str(parse[0])))
                cooked_bw.append(float(str(parse[1])))
        if flag == 1:
            continue
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names

if __name__ == '__main__':
    load_trace('./mrc_div_test/', 0, 1)

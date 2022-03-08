from contextlib import contextmanager


@contextmanager
def write_pruning_rate(pruning_results_file, pruning_rate):
    with open(pruning_results_file, 'w') as file:
        # file.write(pruning_rate)
        file.write("pruning_rate = [\n")
        file.write('\t\t\t'+ str(pruning_rate[0]) + ',  # m.0\n')
        file.write('\t\t\t'+ str(pruning_rate[1]) + ',  # m.1\n')
        file.write('\t\t\t'+ str(pruning_rate[2]) + ',  # m.2\n')

        file.write('\t\t\t')
        for i in range(6):
            file.write(str(pruning_rate[3+i]) + ', ')
        file.write('  # m.3  csp1_1\n')

        file.write('\t\t\t'+ str(pruning_rate[9]) + ',  # m.4\n')

        file.write('\t\t\t')
        for i in range(10):
            file.write(str(pruning_rate[10+i]) + ', ')
        file.write('  # m.5  csp1_3\n')

        file.write('\t\t\t'+ str(pruning_rate[20]) + ',  # m.6\n')

        file.write('\t\t\t')
        for i in range(10):
            file.write(str(pruning_rate[21+i]) + ', ')
        file.write('  # m.7  csp1_3\n')

        file.write('\t\t\t'+ str(pruning_rate[31]) + ',  # m.8\n')

        file.write('\t\t\t')
        for i in range(2):
            file.write(str(pruning_rate[32+i]) + ', ')
        file.write('  # m.9  spp\n')

        file.write('\t\t\t')
        for i in range(6):
            file.write(str(pruning_rate[34+i]) + ', ')
        file.write('  # m.10  csp2_1\n')

        file.write('\t\t\t'+ str(pruning_rate[40]) + ',  # m.11\n')

        file.write('\t\t\t')
        for i in range(6):
            file.write(str(pruning_rate[41+i]) + ', ')
        file.write('  # m.14  csp2_1\n')

        file.write('\t\t\t'+ str(pruning_rate[47]) + ',  # m.15\n')

        file.write('\t\t\t')
        for i in range(6):
            file.write(str(pruning_rate[48+i]) + ', ')
        file.write('  # m.18  csp2_1\n')

        file.write('\t\t\t'+ str(pruning_rate[54]) + ',  # m.19\n')

        file.write('\t\t\t')
        for i in range(6):
            file.write(str(pruning_rate[55+i]) + ', ')
        file.write('  # m.21  csp2_1\n')

        file.write('\t\t\t'+ str(pruning_rate[61]) + ',  # m.22\n')

        file.write('\t\t\t')
        for i in range(6):
            file.write(str(pruning_rate[62+i]) + ', ')
        file.write('  # m.24  csp2_1\n')
        file.write('\t\t\t]\n\n')
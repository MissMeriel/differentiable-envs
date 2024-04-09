# visualizing robust oracle evaluations with different oracle config parameters

from select_grasp import *

if __name__ == "__main__":

    Grasp.logger.info("Running oracle_vis.py")

    files = ["example-grasps/grasp_1.json", "example-grasps/grasp_3.json", "example-grasps/grasp_7.json", "example-grasps/grasp_8.json", ]
    gfiles = [str_ for str_ in files for _ in range(16)]

    # grasp_files, notfound = [], 0
    # for i in range(10):
    #     grasp_file = "example-grasps/grasp_" + str(i) + ".json"
    #     if os.path.isfile(grasp_file):
    #         grasp_files.append(grasp_file)
    #     else:
    #         Grasp.logger.error(f"Grasp file {grasp_file} does not exist.")
    #         notfound += 1

    # gb = Grasp.read_batch(grasp_files)
    # assert len(grasp_files) - notfound == gb.num_grasps()

    # for i, grasp in enumerate(gb):
    #     print(f"Grasp {i}: quality {grasp.quality.item()}")

    gb = Grasp.read_batch(gfiles)
    assert len(gfiles) == gb.num_grasps()

    start = time.time()
    oracle_quals = gb.oracle_eval("data/new_barclamp.obj")
    end1 = time.time()
    oracle_quals_nr = gb.oracle_eval("data/new_barclamp.obj", robust=False)
    end2 = time.time() 
    
    oracle_quals = oracle_quals.squeeze().detach().cpu().numpy().tolist()
    oracle_quals_nr = oracle_quals_nr.squeeze().detach().cpu().numpy().tolist()
    robust_time = end1 - start
    nonrobust_time = end2 - end1
    
    x_values = [0, 1, 2, 3]
    x_labels = ["grasp1", "grasp3", "grasp7", "grasp8"]
    total_points = [oracle_quals[0:16], oracle_quals[16:32], oracle_quals[32:48], oracle_quals[48:]]
    assert len(total_points) == 4
    nr_points = [oracle_quals_nr[0:16], oracle_quals_nr[16:32], oracle_quals_nr[32:48], oracle_quals_nr[48:]]
    nr_points2 = [oracle_quals_nr[0], oracle_quals_nr[16], oracle_quals_nr[32], oracle_quals_nr[48]]
    
    plt.figure(figsize=(9, 9))
    for i, color1_set in enumerate(total_points):
        if i == 0:
            plt.scatter([x_values[i]]*len(color1_set), color1_set, color='red', label='robust')
        else:
            plt.scatter([x_values[i]]*len(color1_set), color1_set, color='red')
            
    for i, color2_set in enumerate(nr_points):
        if i == 0:
            plt.scatter([x_values[i]]*len(color2_set), color2_set, color='blue', label='non-robust')
        else:
            plt.scatter([x_values[i]]*len(color2_set), color2_set, color='blue')

    # standard deviation lines
    means = np.array(total_points).mean(axis=1)
    std_devs = np.array(total_points).std(axis=1)
    plt.errorbar(x_values, means, yerr=std_devs, fmt='o', color='lightgray', ecolor='lightgray', elinewidth=2, capsize=0, label='Standard Deviation')
    for i, txt in enumerate(std_devs):
        plt.text(x_values[i], means[i] + std_devs[i], f'stdev: {txt:.4f}', ha='center',)
        # print(f"x: {x_values[i]}, y: {means[i] + std_devs[i]}, value: {txt:.4f}, non-rounded: {txt}")
 
    plt.xlabel('Grasp')
    plt.xticks(x_values, x_labels, rotation=45)
    plt.ylabel('Oracle Quality')
    plt.title("No friction, translation, or object uncertainty set")
    plt.legend()
    # plt.tight_layout()
    # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.savefig("oracle_vis/no-friction-trans-obj.png")

    data_dict = {}
    data_dict["grasp-files"] = files
    data_dict["robust-quals"] = total_points
    data_dict["nonrobust-quals"] = nr_points2
    data_dict["robust-time"] = robust_time
    data_dict["nonrobust-time"] = nonrobust_time
    with open("oracle_vis/no-friction-trans-obj.txt", "w") as f:
        json.dump(data_dict, f, indent=4)

    Grasp.logger.info("Done running oracle_vis.py")
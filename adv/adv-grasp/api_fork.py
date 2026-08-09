import h5py

class dexnet_db:
    def __init__(self,path):
        self.data_ = h5py.File(path, 'r')


if __name__ == "__main__":
    db = dexnet_db('/mnt/array/Home/Data/HPSTA/dexnet_database/dexnet_2.0_training_database/dexnet_2_database.hdf5')

    datasets = db.data_['datasets']
    print(datasets.keys())
    for dataset in datasets:
        
        for object in db.data_['datasets'][dataset]['objects']:
            inner = db.data_['datasets'][dataset]['objects'][object]
            for name in db.data_['datasets'][dataset]['objects'][object]['grasps']:
                for grasp in db.data_['datasets'][dataset]['objects'][object]['grasps'][name]:
                    if db.data_['datasets'][dataset]['objects'][object]['grasps'][name][grasp]['metrics'].attrs['robust_ferrari_canny'] > 0.8:
                        print('success?')
                        verts = db.data_['datasets'][dataset]['objects'][object]['mesh']['vertices'][:]
                        tris = db.data_['datasets'][dataset]['objects'][object]['mesh']['triangles'][:]
                        print(tris)
    print(db.data_)


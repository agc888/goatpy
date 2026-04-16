import pandas as pd
from pathlib import Path


def annotate_glycans(sdata, glycan_list = None, threshold = 0.2,adata_slot = "maldi_adata" ):
    
    adata = sdata[adata_slot].copy()
    names = []



    if glycan_list is None:
        glycans = pd.read_csv("data/glycan_list.csv")
    else:
        path = Path(glycan_list)

        # 1. Check it exists and is a file
        if not path.exists() or not path.is_file():
            raise ValueError(f"{glycan_list} is not a valid file path")

        # 2. Check it's a CSV
        if path.suffix.lower() != ".csv":
            raise ValueError(f"{glycan_list} is not a CSV file")

        # 3. Try reading it
        glycans = pd.read_csv(path)

        # 4. Check it has exactly two columns
        if glycans.shape[1] != 2:
            raise ValueError(
                f"{glycan_list} must contain exactly 2 columns, found {glycans.shape[1]}"
            )
        


    for mz in adata.var_names:
        gly_name = []
        mz = np.float64(mz)#.split("-")[1])

        upper = mz + threshold  
        lower = mz - threshold

        for idx, x in enumerate(np.array(glycans.iloc[:,0])):
            if x >= lower and x <= upper:
                gly_name.append(np.array(glycans.iloc[:,1])[idx])

        if len(gly_name) > 1:
            names.append((", ".join(gly_name)))
        elif len(gly_name) < 1:
            names.append("mz-"+str(mz))
        else:
            names.append(gly_name[0])
        
    adata.var_names = names

    sdata[adata_slot] = adata
    return(sdata)
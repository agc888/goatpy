import goatpy as gp


path = "/Users/andrewcauser/Documents/Griffith/AD_resolution_change_check_10um_actual_10012020_2.imzML"
he_path = "/Users/andrewcauser/Documents/Griffith/res_check_0000.tif"

#path = "/Users/andrewcauser/Documents/Griffith/AD_resolution_change_check_10um_actual_10012020_2.imzML"
#he_path = "/Users/andrewcauser/Documents/Griffith/res_check_0000.tif"



test = gp.io.glyco_spatialdata(imzml_path=path)
test = gp.Add_Pseudo_Image(test, "TIC", library_id = "Spatial")

he_test = gp.he_spatialdata(he_path)



from napari_spatialdata import Interactive

interactive = Interactive([test, he_test])
interactive.run()



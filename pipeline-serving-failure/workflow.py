from kfp import dsl
import mlrun
import os
@dsl.pipeline(name="naipi pipeline")
def kfpipeline():
	project = mlrun.get_current_project()
	function_rbtlafridy = project.get_function("nrqhh")
	function_build = mlrun.build_function(function_rbtlafridy)
	function_output = mlrun.run_function(function_rbtlafridy, handler="aihubqqfxgdppdikwzya", outputs=["clf_model"]).after(function_build)
	project.get_function("serving").add_model(key = "my_model",
		class_name = "bad_class",
		model_path = function_output.outputs["clf_model"])
	project.get_function("serving").apply(mlrun.auto_mount())
	mlrun.deploy_function(project.get_function("serving"))

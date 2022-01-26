from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
from wandb_api_key import WANDB_API_KEY

if __name__ == '__main__':
    # Pip packages
    with open('requirements.txt', 'r') as f:
        pip_packages = [
            pkg.strip().lstrip('-e ') for pkg in f if not pkg.startswith('#')]

    # Workspace and environment
    ws = Workspace.from_config()
   
    env = Environment(name='mnist-env')
    env.environment_variables = {'WANDB_API_KEY': WANDB_API_KEY}
    
    conda_dep = CondaDependencies.create(
        python_version='3.7.2',
        conda_packages=['pip==21.1.2'],
        pip_packages=pip_packages)
    env.python.conda_dependencies = conda_dep

    # Create a script config
    script_config = ScriptRunConfig(
        source_directory='.',
        script='src/models/train_model.py',
        environment=env,
        compute_target='MLOpsTest')
    
    # Run experiment
    experiment = Experiment(workspace=ws, name='mnist-test')
    run = experiment.submit(script_config)

    run.wait_for_completion()
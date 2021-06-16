from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
from wandb_api_key import WANDB_API_KEY
#from azureml.widgets import RunDetails

if __name__ == '__main__':
    # Pip packages
    with open('requirements.txt', 'r') as f:
        pip_packages = [
            pkg.strip().lstrip('-e ') for pkg in f if not pkg.startswith('#')]
    pip_packages.remove('git+https://github.com/jonasvj/MLOps@d8af0044cc5bcd98a81e4128ac13b169cd619b97#egg=src')
    print(pip_packages)
    # Setup workspace and environment
    ws = Workspace.from_config()
   
    env = Environment(name='mnist-env')
    env.environment_variables = {'WANDB_API_KEY': WANDB_API_KEY}
    
    conda_dep = CondaDependencies.create(
        python_version='3.7.2',
        conda_packages=['pip==21.1.2'],
        pip_packages=pip_packages)
    conda_dep.set_pip_option('-e git+https://github.com/jonasvj/MLOps@d8af0044cc5bcd98a81e4128ac13b169cd619b97#egg=src')
    #conda_dep.set_pip_option('-e .')
    env.python.conda_dependencies = conda_dep
    #env.python.user_managed_dependencies = True

    # Create a script config
    script_config = ScriptRunConfig(
        source_directory='.',
        script='src/models/train_model.py',
        environment=env,
        compute_target='MLOpsTest')
    
    # Run experiment
    experiment = Experiment(workspace=ws, name='mnist-test')
    run = experiment.submit(script_config)

    #RunDetails(run).show()
    run.wait_for_completion()
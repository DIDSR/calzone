### This small section is used to help with the documentation only and are not part of calzone.


from calzone.utils import fake_binary_data_generator
import pandas as pd
import numpy as np

def generate_wellcal_data(sample_size, save_path, alpha_val=0.5, beta_val=0.5, random_seed=123):
    """
    Generate fake well-calibrated data for demonstration purposes.

    Args:
        sample_size (int): The number of samples to generate.
        save_path (str): The file path to save the generated data.
        alpha_val (float, optional): Alpha parameter for the beta distribution. Defaults to 0.5.
        beta_val (float, optional): Beta parameter for the beta distribution. Defaults to 0.5.
        random_seed (int, optional): Seed for random number generation. Defaults to 123.

    Returns:
        None
    """
    np.random.seed(random_seed)
    fakedata_generator = fake_binary_data_generator(alpha_val=alpha_val, beta_val=beta_val)
    X, y_true = fakedata_generator.generate_data(sample_size)
    df = np.column_stack((np.array(X).tolist(), np.array(y_true).tolist()))
    np.savetxt(save_path, df, delimiter=",", header="proba_0,proba_1,label", comments='')
    return 

def generate_miscal_data(sample_size, save_path, miscal_scale, alpha_val=0.5, beta_val=0.5, random_seed=123):
    """
    Generate fake miscalibrated data for demonstration purposes.

    Args:
        sample_size (int): The number of samples to generate.
        save_path (str): The file path to save the generated data.
        miscal_scale (float): The scale of miscalibration to apply.
        alpha_val (float, optional): Alpha parameter for the beta distribution. Defaults to 0.5.
        beta_val (float, optional): Beta parameter for the beta distribution. Defaults to 0.5.
        random_seed (int, optional): Seed for random number generation. Defaults to 123.

    Returns:
        None
    """
    np.random.seed(random_seed)
    fakedata_generator = fake_binary_data_generator(alpha_val=alpha_val, beta_val=beta_val)
    X, y_true = fakedata_generator.generate_data(sample_size)
    X = fakedata_generator.linear_miscal(X, miscal_scale=miscal_scale)
    df = np.column_stack((np.array(X).tolist(), np.array(y_true).tolist()))
    np.savetxt(save_path, df, delimiter=",", header="proba_0,proba_1,label", comments='')
    return

def generate_subgroup_data(sample_size, save_path, miscal_scale, alpha_val=0.5, beta_val=0.5, random_seed=123):
    """
    Generate fake data with subgroups for demonstration purposes.

    Args:
        sample_size (int): The number of samples to generate for each subgroup.
        save_path (str): The file path to save the generated data.
        miscal_scale (float): The scale of miscalibration to apply to subgroup B.
        alpha_val (float, optional): Alpha parameter for the beta distribution. Defaults to 0.5.
        beta_val (float, optional): Beta parameter for the beta distribution. Defaults to 0.5.
        random_seed (int, optional): Seed for random number generation. Defaults to 123.

    Returns:
        None
    """
    np.random.seed(random_seed)
    fakedata_generator = fake_binary_data_generator(alpha_val=alpha_val, beta_val=beta_val)
    X, y_true = fakedata_generator.generate_data(sample_size)
    X2, y_true2 = fakedata_generator.generate_data(sample_size)
    X2 = fakedata_generator.linear_miscal(X2, miscal_scale=miscal_scale)
    df_A = pd.DataFrame({'proba_0': X[:, 0], 'proba_1': X[:, 1], 'subgroup': 'A', 'label': y_true})
    df_B = pd.DataFrame({'proba_0': X2[:, 0], 'proba_1': X2[:, 1], 'subgroup': 'B', 'label': y_true2})
    df_merged = pd.concat([df_A, df_B], ignore_index=True)
    # Rename the 'subgroup' column to 'subgroup_1'
    df_merged = df_merged.rename(columns={'subgroup': 'subgroup_1'})
    df_merged.to_csv(save_path, index=False)
    return 
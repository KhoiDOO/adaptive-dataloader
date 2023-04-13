import deeplake

class DlakeInterface:
    def __init__(self) -> None:
        pass
        # self.available_dataset = deeplake
    
    def __check_dataset_name(self, dataset_name:str = None, subset:str = None):
        if dataset_name is None:
            raise Exception("dataset_name cannot be None")
        elif not isinstance(dataset_name, str):
            raise Exception(f"dataset_name must be a string but found {type(dataset_name)} instead")
        # elif f"{dataset_name}-{subset}" not in self.available_dataset:
        #     raise Exception(f"The dataset {dataset_name}-{subset} is not currently support")
    
    def __check_subset(self, subset:str = None):
        if subset is None:
            raise Exception("subset cannot be None")
        elif not isinstance(subset, str):
            raise Exception(f"subset must be a string but found {type(subset)} instead")
    
    def __check_export(self, export:str = None):
        if export is None:
            raise Exception("export cannot be None")
        elif not isinstance(export, str):
            raise Exception(f"export must be a string but found {type(export)} instead")
    
    def get_dataset(self, dataset_name, subset) -> deeplake.Dataset:
        self.__check_dataset_name(dataset_name=dataset_name)
        self.__check_subset(subset=subset)
        
        return deeplake.load(f'hub://activeloop/{dataset_name}-{subset}')
    
    def get_torch_dataloader(self, 
                       dataset_name, 
                       subset,
                       *args, **kwargs):
        ds = self.get_dataset(dataset_name=dataset_name, subset=subset)
        
        return ds.pytorch(*args, **kwargs)

if __name__ == "__main__":
    interface = DlakeInterface()
    
    dataloader = interface.get_torch_dataloader(
        dataset_name="mnist", 
        subset="test",
        shuffle = True,
        decode_method = {"images" : "pil"})
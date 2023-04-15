import deeplake

class DlakeInterface:
    def __init__(self) -> None:
        pass
        # self.available_dataset = deeplake
    
    def __check_dataset_name(self, dataset_name:str = None, subset:str = None):
        if dataset_name is None:
            raise ValueError("dataset_name cannot be None")
        elif not isinstance(dataset_name, str):
            raise TypeError(f"dataset_name must be a string but found {type(dataset_name)} instead")
    
    def __check_subset(self, subset:str = None):
        if subset is None:
            raise ValueError("subset cannot be None")
        elif not isinstance(subset, str):
            raise TypeError(f"subset must be a string but found {type(subset)} instead")
        elif subset not in ["train", "val", "test"]:
            raise Exception(f"subset must be train, val or test but found {type(subset)} instead")
        
    def __check_access_method(self, access_method:str = None):
        if access_method is None:
            raise ValueError("access_method cannot be None")
        elif not isinstance(access_method, str):
            raise TypeError(f"access_method must be a string but found {type(access_method)} instead") 
        elif access_method not in ["stream", "local", "download"]:
            raise Exception(f"access_method must be stream, local or download but found {type(access_method)} instead")
    
    def get_dataset(self, 
                    dataset_name:str = None, 
                    subset:str = None, 
                    access_method:str = None) -> deeplake.Dataset:
        self.__check_dataset_name(dataset_name=dataset_name)
        self.__check_subset(subset=subset)
        self.__check_access_method(access_method=access_method)
        
        return deeplake.load(
            path = f'hub://activeloop/{dataset_name}-{subset}',
            access_method = access_method)
    
    def get_torch_dataloader(self, 
                       dataset_name, 
                       subset,
                       access_method,
                       *args, **kwargs):
        ds = self.get_dataset(
            dataset_name=dataset_name, 
            subset=subset,
            access_method = access_method)
        
        return ds.pytorch(*args, **kwargs)

if __name__ == "__main__":
    interface = DlakeInterface()
    
    dataloader = interface.get_torch_dataloader(
        dataset_name="mnist", 
        subset="test",
        shuffle = True,
        decode_method = {"images" : "pil"})
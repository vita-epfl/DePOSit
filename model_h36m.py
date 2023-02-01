from model_base import ModelBase


class Model_H36M(ModelBase):
    def __init__(self, config, device, target_dim=96):
        super(Model_H36M, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        pose = batch["pose"].to(self.device).float()
        tp = batch["timepoints"].to(self.device).float()
        mask = batch["mask"].to(self.device).float()

        pose = pose.permute(0, 2, 1)
        mask = mask.permute(0, 2, 1)

        return (
            pose,
            tp,
            mask
        )

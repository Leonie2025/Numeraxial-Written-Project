from dataclasses import dataclass
import math
@dataclass
class SequentialMacroKalman:
    mean: float
    var: float
    process_var: float=0.0025
    def predict(self): self.var += self.process_var; return self.mean,self.var
    def update(self, observation: float, obs_var: float):
        k=self.var/(self.var+obs_var); self.mean=self.mean+k*(observation-self.mean); self.var=(1-k)*self.var; return self.mean,self.var,k
    def update_ofi_text(self, ofi_rate_obs: float, ofi_var: float, text_rate_obs: float, text_var: float):
        self.predict(); a=self.update(ofi_rate_obs,ofi_var); b=self.update(text_rate_obs,text_var); return {'posterior_mean':self.mean,'posterior_sd':math.sqrt(self.var),'k_ofi':a[2],'k_text':b[2]}

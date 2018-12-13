AL_MAPPING = {}

from sampling_methods.margin_AL import MarginAL
from sampling_methods.uniform_sampling import UniformSampling
AL_MAPPING['margin'] = MarginAL
AL_MAPPING['uniform'] = UniformSampling

def get_AL_sampler(name):
  if name in AL_MAPPING and name != 'mixture_of_samplers':
    return AL_MAPPING[name]
  if 'mixture_of_samplers' in name:
    return get_mixture_of_samplers(name)
  raise NotImplementedError('The specified sampler is not available.')


def get_mixture_of_samplers(name):
    return None
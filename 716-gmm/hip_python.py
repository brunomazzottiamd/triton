# -*- coding: utf-8 -*-


# https://rocm.docs.amd.com/projects/hip-python/en/latest/index.html
# https://rocm.docs.amd.com/projects/hip-python/en/latest/user_guide/1_usage.html#via-hipgetdeviceproperties


from hip import hip


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def num_sms(device: int = 0) -> int:
    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, device))
    assert hasattr(props, "multiProcessorCount"), "Could not find device property."
    num_sms = int(getattr(props, "multiProcessorCount"))
    assert num_sms > 0, f"Number of SMs must be positive (it's {num_sms})."
    return num_sms


if __name__ == "__main__":
    print(f"We have {num_sms()} SMs.")

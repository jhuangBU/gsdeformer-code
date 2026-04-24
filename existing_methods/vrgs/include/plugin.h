#pragma once
#include "Unity/IUnityInterface.h"
#include "Unity/IUnityGraphics.h"

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginUnload();
extern "C" UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API GetRenderEventFunc();

extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsAPIReady();
extern "C" UNITY_INTERFACE_EXPORT const char* UNITY_INTERFACE_API GetLastMessage();
extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API LoadModel(const char* filepath);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetNbPov(int nb_pov);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetPovParameters(int pov, int width, int height);
extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsInitialized();
extern "C" UNITY_INTERFACE_EXPORT void* UNITY_INTERFACE_API GetTextureNativePointer(int pov);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetDrawParameters(int pov, float* position, float* rotation, float* proj, float fovy, float* frustums);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetLightingParameters(int H, int W, float* position, float* rotation, float* proj, float fovy, float* frustums);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API GetSceneSize(float* scene_min, float* scene_max);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetSimParams(
    float frame_dt, 
    float dt, 
    float gravity, 
    float damping_coeffient,
    int rest_iter, 
    float collision_stiffness, 
    float minimal_dist, 
    float collision_detection_iter_interval, 
    float shadow_eps,
    float shadow_factor,
    int zup, 
    float ground_height, 
    float global_scale, 
    float *global_offset, 
    float *global_rotation, 
    int num_objects, 
    float *object_offsets, 
    float *object_materials,
    int disable_LG_interp,
    int quasi_static,
    int max_quasi_sequence,
    int enable_export_frames,
    int max_export_sequence,
    int boundary);
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetController(
    float controller_radius,
    float *lpos, float *lrot, int ltriggered,
	float *rpos, float *rrot, int rtriggered);
extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API DrawSync();
extern "C" UNITY_INTERFACE_EXPORT bool UNITY_INTERFACE_API IsDrawn();

extern "C" UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API GetNbSplat();
extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API Save();
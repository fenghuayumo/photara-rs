/* SPDX-License-Identifier: Apache-2.0 */
/* Included in the independently named Photara Python module only. */
#ifdef WITH_DLSS
uint64_t dlss_evaluation_count();
#endif
static PyObject *photara_realtime_stats_func(PyObject *, PyObject *handle)
{
#ifdef WITH_PHOTARA_REALTIME
  auto *owner = static_cast<BlenderSession *>(PyLong_AsVoidPtr(handle));
  if (PyErr_Occurred()) return nullptr;
  if (!owner || !owner->session || owner->session->is_rendering()) Py_RETURN_NONE;
  auto &history = owner->realtime_history;
  history.sync_stats();
  return Py_BuildValue("{si,si,si,sK,sK,sK,sK}", "stored_frames", history.stored_frames(),
                       "samples", owner->session->params.samples,
                       "frame", owner->realtime_last_frame,
                       "gpu_updates", (unsigned long long)history.gpu_updates,
                       "reused_pixels", (unsigned long long)history.reused_pixels,
                       "max_history_frames", (unsigned long long)history.reused_frames,
                       "reconstruction_pixel_readback_bytes", 0ULL);
#else
  Py_RETURN_NONE;
#endif
}
static PyObject *photara_session_stats_func(PyObject *, PyObject *handle)
{
  auto *owner = static_cast<BlenderSession *>(PyLong_AsVoidPtr(handle));
  if (PyErr_Occurred()) {
    return nullptr;
  }
  if (!owner || !owner->session || !owner->scene) {
    Py_RETURN_NONE;
  }
  Scene *scene = owner->scene;
  if (!scene->mutex.try_lock()) {
    Py_RETURN_NONE;
  }
  const thread_scoped_lock lock(scene->mutex, std::adopt_lock);
  unsigned long long nodes = 0, images = 0;
  for (const Shader *shader : scene->shaders) {
    if (shader->graph) {
      for (const ShaderNode *node : shader->graph->nodes) {
        ++nodes;
        images += node->type->name == ustring("image_texture");
      }
    }
  }
  const auto &params = owner->session->params;
  uint64_t dlss_evaluations = 0;
#ifdef WITH_DLSS
  dlss_evaluations = dlss_evaluation_count();
#endif
  return Py_BuildValue("{sK,sO,si,sd,sO,sO,si,si,si,si,si,si,si,sd,sK,sK,sK,sK,sK,sK}",
                       "dlss_evaluations", (unsigned long long)dlss_evaluations,
                       "use_denoise", scene->integrator->get_use_denoise() ? Py_True : Py_False,
                       "denoiser_type", int(scene->integrator->get_denoiser_type()),
                       "denoiser_upscale_factor", double(scene->integrator->get_denoiser_upscale_factor()),
                       "use_resolution_divider", params.use_resolution_divider ? Py_True : Py_False,
                       "stable_planes", scene->dscene.data.film.pass_stable_planes != PASS_UNUSED ?
                                            Py_True : Py_False,
                       "pixel_size", params.pixel_size,
                       "trace_width", int(scene->dscene.data.cam.width),
                       "trace_height", int(scene->dscene.data.cam.height),
                       "neeat_emitters", scene->dscene.data.integrator.neeat_count,
                       "neeat_local_tables", scene->dscene.data.integrator.neeat_local_count,
                       "neeat_clusters", scene->dscene.data.integrator.neeat_cluster_count,
                       "neeat_lod_stride", scene->dscene.data.integrator.neeat_cluster_size,
                       "neeat_tree_fraction", double(scene->dscene.data.integrator.neeat_tree_fraction),
                       "neeat_device_bytes", (unsigned long long)(scene->dscene.neeat_lights.size() * sizeof(KernelNEEATLight) +
                           scene->dscene.neeat_lookup.size() * sizeof(KernelNEEATLookup) +
                           scene->dscene.neeat_clusters.size() * sizeof(KernelNEEATCluster) +
                           scene->dscene.neeat_distribution.size() * sizeof(float2) +
                           scene->dscene.neeat_feedback.size() * sizeof(float)),
                       "light_tree_emitters", (unsigned long long)scene->dscene.light_tree_emitters.size(),
                       "objects", (unsigned long long)scene->objects.size(),
                       "shaders", (unsigned long long)scene->shaders.size(),
                       "shader_nodes", nodes,
                       "image_texture_nodes", images);
}

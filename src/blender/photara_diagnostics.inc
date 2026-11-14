/* SPDX-License-Identifier: Apache-2.0 */
/* Included in the independently named Photara Python module only. */
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
                       "reconstruction_pixel_readback_bytes", (unsigned long long)history.pixel_readback_bytes);
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
  return Py_BuildValue("{sO,si,si,si,sK,sK,sK,sK}",
                       "use_resolution_divider", params.use_resolution_divider ? Py_True : Py_False,
                       "pixel_size", params.pixel_size,
                       "trace_width", int(scene->dscene.data.cam.width),
                       "trace_height", int(scene->dscene.data.cam.height),
                       "objects", (unsigned long long)scene->objects.size(),
                       "shaders", (unsigned long long)scene->shaders.size(),
                       "shader_nodes", nodes,
                       "image_texture_nodes", images);
}

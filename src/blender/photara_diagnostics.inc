/* SPDX-License-Identifier: Apache-2.0 */
/* Included in the independently named Photara Python module only. */
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

/* SPDX-License-Identifier: Apache-2.0 */
#pragma once
#define VK_USE_PLATFORM_WIN32_KHR
#include "util/path.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>

/* CUDA completes film conversion before DisplayDriver::update_end. The shared
 * mutex and Vulkan fence return exclusive ownership before the next CUDA write.
 * This intentionally bounded baseline uses host completion, not external GPU
 * timeline semaphores; the pixel data itself never passes through CPU memory. */
class ViewerPresenter {
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t family_ = 0, size_ = 0;
  HWND window_ = nullptr;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VkDeviceSize allocation_ = 0;
  VkSwapchainKHR swap_ = VK_NULL_HANDLE;
  VkFormat format_ = VK_FORMAT_B8G8R8A8_UNORM;
  VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
  VkExtent2D extent_ = {};
  VkRenderPass pass_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptors_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkDescriptorSet set_ = VK_NULL_HANDLE;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkCommandBuffer command_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  VkSemaphore acquired_ = VK_NULL_HANDLE;
  std::vector<VkImageView> views_;
  std::vector<VkFramebuffer> frames_;
  std::vector<VkSemaphore> presented_;
  std::vector<uint8_t> uuid_;
  bool resized_ = true;

  static void check(VkResult r)
  {
    if (r != VK_SUCCESS)
      throw std::runtime_error("Vulkan viewer error: " + std::to_string(r));
  }
  uint32_t memory_type(uint32_t mask, VkMemoryPropertyFlags flags)
  {
    VkPhysicalDeviceMemoryProperties p;
    vkGetPhysicalDeviceMemoryProperties(physical_, &p);
    for (uint32_t i = 0; i < p.memoryTypeCount; ++i)
      if ((mask & (1u << i)) && (p.memoryTypes[i].propertyFlags & flags) == flags)
        return i;
    throw std::runtime_error("No suitable Vulkan memory type");
  }
  VkShaderModule shader(const char *name)
  {
    std::ifstream file(ccl::path_get(name), std::ios::binary | std::ios::ate);
    if (!file)
      throw std::runtime_error(std::string("Missing SPIR-V shader: ") + name);
    const size_t bytes = size_t(file.tellg());
    if (!bytes || bytes % 4)
      throw std::runtime_error("Invalid SPIR-V size");
    std::vector<uint32_t> code(bytes / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char *>(code.data()), bytes);
    if (!file)
      throw std::runtime_error("Cannot read SPIR-V");
    VkShaderModuleCreateInfo info = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = bytes;
    info.pCode = code.data();
    VkShaderModule module;
    check(vkCreateShaderModule(device_, &info, nullptr, &module));
    return module;
  }
  void destroy_swap()
  {
    for (auto f : frames_)
      vkDestroyFramebuffer(device_, f, nullptr);
    for (auto v : views_)
      vkDestroyImageView(device_, v, nullptr);
    for (auto s : presented_)
      vkDestroySemaphore(device_, s, nullptr);
    frames_.clear();
    views_.clear();
    presented_.clear();
    if (swap_)
      vkDestroySwapchainKHR(device_, swap_, nullptr);
    swap_ = VK_NULL_HANDLE;
  }
  bool resize()
  {
    RECT rect;
    GetClientRect(window_, &rect);
    if (IsIconic(window_) || rect.right < 1 || rect.bottom < 1)
      return false;
    VkSurfaceCapabilitiesKHR caps;
    check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface_, &caps));
    auto extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
      extent.width = std::clamp(
          uint32_t(rect.right), caps.minImageExtent.width, caps.maxImageExtent.width);
      extent.height = std::clamp(
          uint32_t(rect.bottom), caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    if (!resized_ && extent.width == extent_.width && extent.height == extent_.height)
      return true;
    check(vkDeviceWaitIdle(device_));
    destroy_swap();
    uint32_t count = std::max(3u, caps.minImageCount);
    if (caps.maxImageCount)
      count = std::min(count, caps.maxImageCount);
    VkSwapchainCreateInfoKHR info = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = surface_;
    info.minImageCount = count;
    info.imageFormat = format_;
    info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    info.imageExtent = extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = caps.currentTransform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = present_mode_;
    info.clipped = VK_TRUE;
    check(vkCreateSwapchainKHR(device_, &info, nullptr, &swap_));
    check(vkGetSwapchainImagesKHR(device_, swap_, &count, nullptr));
    std::vector<VkImage> images(count);
    check(vkGetSwapchainImagesKHR(device_, swap_, &count, images.data()));
    for (auto image : images) {
      VkImageViewCreateInfo view = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view.image = image;
      view.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view.format = format_;
      view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      VkImageView v;
      check(vkCreateImageView(device_, &view, nullptr, &v));
      views_.push_back(v);
      VkFramebufferCreateInfo frame = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      frame.renderPass = pass_;
      frame.attachmentCount = 1;
      frame.pAttachments = &views_.back();
      frame.width = extent.width;
      frame.height = extent.height;
      frame.layers = 1;
      VkFramebuffer f;
      check(vkCreateFramebuffer(device_, &frame, nullptr, &f));
      frames_.push_back(f);
      VkSemaphoreCreateInfo sem = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      VkSemaphore s;
      check(vkCreateSemaphore(device_, &sem, nullptr, &s));
      presented_.push_back(s);
    }
    extent_ = extent;
    resized_ = false;
    return true;
  }
  void ownership(bool acquire,
                 VkAccessFlags access = VK_ACCESS_SHADER_READ_BIT,
                 VkPipelineStageFlags stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
  {
    VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = acquire ? 0 : access;
    barrier.dstAccessMask = acquire ? access : 0;
    barrier.srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_EXTERNAL : family_;
    barrier.dstQueueFamilyIndex = acquire ? family_ : VK_QUEUE_FAMILY_EXTERNAL;
    barrier.buffer = buffer_;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(command_,
                         acquire ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : stage,
                         acquire ? stage : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0,
                         0,
                         nullptr,
                         1,
                         &barrier,
                         0,
                         nullptr);
  }
  void begin()
  {
    check(vkResetCommandBuffer(command_, 0));
    VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command_, &begin));
  }

 public:
  std::mutex interop_mutex;
  const std::vector<uint8_t> &device_uuid() const
  {
    return uuid_;
  }
  size_t buffer_bytes() const
  {
    return size_t(allocation_);
  }
  void wait_for_reader()
  {
    check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX));
  }
  HANDLE export_handle()
  {
    auto get = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandleKHR"));
    if (!get)
      throw std::runtime_error("Vulkan Win32 memory export unavailable");
    VkMemoryGetWin32HandleInfoKHR info = {VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    info.memory = memory_;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE handle;
    check(get(device_, &info, &handle));
    return handle;
  }
  ~ViewerPresenter()
  {
    if (device_) {
      vkDeviceWaitIdle(device_);
      destroy_swap();
      if (pipeline_)
        vkDestroyPipeline(device_, pipeline_, nullptr);
      if (layout_)
        vkDestroyPipelineLayout(device_, layout_, nullptr);
      if (descriptors_)
        vkDestroyDescriptorPool(device_, descriptors_, nullptr);
      if (set_layout_)
        vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
      if (pass_)
        vkDestroyRenderPass(device_, pass_, nullptr);
      if (acquired_)
        vkDestroySemaphore(device_, acquired_, nullptr);
      if (fence_)
        vkDestroyFence(device_, fence_, nullptr);
      if (pool_)
        vkDestroyCommandPool(device_, pool_, nullptr);
      if (buffer_)
        vkDestroyBuffer(device_, buffer_, nullptr);
      if (memory_)
        vkFreeMemory(device_, memory_, nullptr);
      vkDestroyDevice(device_, nullptr);
    }
    if (surface_)
      vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (instance_)
      vkDestroyInstance(instance_, nullptr);
  }
  void initialize(HWND window, int size)
  {
    window_ = window;
    size_ = size;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Photara Viewer";
    app.apiVersion = VK_API_VERSION_1_1;
    const char *extensions[] = {VK_KHR_SURFACE_EXTENSION_NAME,
                                VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    VkInstanceCreateInfo instance = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance.pApplicationInfo = &app;
    instance.enabledExtensionCount = 2;
    instance.ppEnabledExtensionNames = extensions;
    check(vkCreateInstance(&instance, nullptr, &instance_));
    VkWin32SurfaceCreateInfoKHR surface = {VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    surface.hinstance = GetModuleHandle(nullptr);
    surface.hwnd = window;
    check(vkCreateWin32SurfaceKHR(instance_, &surface, nullptr, &surface_));
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()));
    for (auto physical : devices) {
      VkPhysicalDeviceIDProperties id = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
      VkPhysicalDeviceProperties2 props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
      props.pNext = &id;
      vkGetPhysicalDeviceProperties2(physical, &props);
      if (props.properties.vendorID != 0x10de)
        continue;
      uint32_t n = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(physical, &n, nullptr);
      std::vector<VkQueueFamilyProperties> families(n);
      vkGetPhysicalDeviceQueueFamilyProperties(physical, &n, families.data());
      for (uint32_t i = 0; i < n; ++i) {
        VkBool32 supported = VK_FALSE;
        check(vkGetPhysicalDeviceSurfaceSupportKHR(physical, i, surface_, &supported));
        if (supported && (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
          physical_ = physical;
          family_ = i;
          uuid_.assign(id.deviceUUID, id.deviceUUID + VK_UUID_SIZE);
          break;
        }
      }
      if (physical_)
        break;
    }
    if (!physical_)
      throw std::runtime_error("No NVIDIA Vulkan graphics/present queue");
    const float priority = 1;
    VkDeviceQueueCreateInfo queue = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = family_;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    const char *device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                       VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME};
    VkDeviceCreateInfo device = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device.queueCreateInfoCount = 1;
    device.pQueueCreateInfos = &queue;
    device.enabledExtensionCount = 2;
    device.ppEnabledExtensionNames = device_extensions;
    check(vkCreateDevice(physical_, &device, nullptr, &device_));
    vkGetDeviceQueue(device_, family_, 0, &queue_);
    VkPhysicalDeviceExternalBufferInfo external_query = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
    external_query.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    external_query.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkExternalBufferProperties external_properties = {
        VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
    vkGetPhysicalDeviceExternalBufferProperties(physical_, &external_query, &external_properties);
    const auto features = external_properties.externalMemoryProperties.externalMemoryFeatures;
    if (!(features & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) ||
        (features & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT))
      throw std::runtime_error("Shared Vulkan buffer requires exportable non-dedicated memory");
    VkExternalMemoryBufferCreateInfo external = {
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkBufferCreateInfo buffer = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer.pNext = &external;
    buffer.size = VkDeviceSize(size) * size * 8;
    buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    check(vkCreateBuffer(device_, &buffer, nullptr, &buffer_));
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device_, buffer_, &requirements);
    VkExportMemoryAllocateInfo export_info = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    export_info.handleTypes = external.handleTypes;
    VkMemoryAllocateInfo allocate = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.pNext = &export_info;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = memory_type(requirements.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    allocation_ = requirements.size;
    check(vkAllocateMemory(device_, &allocate, nullptr, &memory_));
    check(vkBindBufferMemory(device_, buffer_, memory_, 0));
    VkDescriptorSetLayoutBinding binding = {
        0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo set = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set.bindingCount = 1;
    set.pBindings = &binding;
    check(vkCreateDescriptorSetLayout(device_, &set, nullptr, &set_layout_));
    VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo descriptors = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    descriptors.maxSets = 1;
    descriptors.poolSizeCount = 1;
    descriptors.pPoolSizes = &pool_size;
    check(vkCreateDescriptorPool(device_, &descriptors, nullptr, &descriptors_));
    VkDescriptorSetAllocateInfo descriptor = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    descriptor.descriptorPool = descriptors_;
    descriptor.descriptorSetCount = 1;
    descriptor.pSetLayouts = &set_layout_;
    check(vkAllocateDescriptorSets(device_, &descriptor, &set_));
    VkDescriptorBufferInfo buffer_info = {buffer_, 0, buffer.size};
    VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set_;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &buffer_info;
    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &count, nullptr));
    std::vector<VkSurfaceFormatKHR> formats(count);
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &count, formats.data()));
    uint32_t mode_count = 0;
    check(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface_, &mode_count, nullptr));
    std::vector<VkPresentModeKHR> modes(mode_count);
    check(
        vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface_, &mode_count, modes.data()));
    if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.end())
      present_mode_ = VK_PRESENT_MODE_MAILBOX_KHR;
    bool supported = false;
    for (auto f : formats)
      if (f.format == format_ && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        supported = true;
    if (!supported)
      throw std::runtime_error("BGRA8 UNORM surface format unavailable");
    VkAttachmentDescription attachment = {};
    attachment.format = format_;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference reference = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &reference;
    VkRenderPassCreateInfo pass = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass.attachmentCount = 1;
    pass.pAttachments = &attachment;
    pass.subpassCount = 1;
    pass.pSubpasses = &subpass;
    check(vkCreateRenderPass(device_, &pass, nullptr, &pass_));
    VkPushConstantRange constants = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4};
    VkPipelineLayoutCreateInfo layout = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &set_layout_;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &constants;
    check(vkCreatePipelineLayout(device_, &layout, nullptr, &layout_));
    auto vert = shader("shaders/viewer.vert.spv"), frag = shader("shaders/viewer.frag.spv");
    VkPipelineShaderStageCreateInfo stages[2] = {};
    for (auto &stage : stages) {
      stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stage.pName = "main";
    }
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    VkPipelineVertexInputStateCreateInfo vertex = {
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly = {
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport = {
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster = {
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo samples = {
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend = {};
    blend.colorWriteMask = 15;
    VkPipelineColorBlendStateCreateInfo blending = {
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blending.attachmentCount = 1;
    blending.pAttachments = &blend;
    VkDynamicState states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic = {
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = states;
    VkGraphicsPipelineCreateInfo pipeline = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.stageCount = 2;
    pipeline.pStages = stages;
    pipeline.pVertexInputState = &vertex;
    pipeline.pInputAssemblyState = &assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &samples;
    pipeline.pColorBlendState = &blending;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = layout_;
    pipeline.renderPass = pass_;
    auto result = vkCreateGraphicsPipelines(
        device_, VK_NULL_HANDLE, 1, &pipeline, nullptr, &pipeline_);
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    check(result);
    VkCommandPoolCreateInfo pool = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = family_;
    check(vkCreateCommandPool(device_, &pool, nullptr, &pool_));
    VkCommandBufferAllocateInfo command = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command.commandPool = pool_;
    command.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(device_, &command, &command_));
    VkFenceCreateInfo fence = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    check(vkCreateFence(device_, &fence, nullptr, &fence_));
    VkSemaphoreCreateInfo semaphore = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check(vkCreateSemaphore(device_, &semaphore, nullptr, &acquired_));
    resize();
    // Establish external ownership before CUDA's first write.
    begin();
    ownership(false);
    check(vkEndCommandBuffer(command_));
    check(vkResetFences(device_, 1, &fence_));
    VkSubmitInfo initial = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    initial.commandBufferCount = 1;
    initial.pCommandBuffers = &command_;
    check(vkQueueSubmit(queue_, 1, &initial, fence_));
    wait_for_reader();
  }
  bool present(HWND window)
  {
    std::unique_lock<std::mutex> lock(interop_mutex, std::try_to_lock);
    if (!lock.owns_lock() || IsIconic(window) || vkGetFenceStatus(device_, fence_) == VK_NOT_READY)
      return false;
    if (!resize())
      return false;
    uint32_t index;
    auto result = vkAcquireNextImageKHR(device_, swap_, 0, acquired_, VK_NULL_HANDLE, &index);
    if (result == VK_NOT_READY || result == VK_TIMEOUT)
      return false;
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
      resized_ = true;
      return false;
    }
    if (result != VK_SUBOPTIMAL_KHR)
      check(result);
    begin();
    ownership(true);
    VkClearValue clear = {};
    clear.color.float32[3] = 1;
    VkRenderPassBeginInfo pass = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass = pass_;
    pass.framebuffer = frames_[index];
    pass.renderArea.extent = extent_;
    pass.clearValueCount = 1;
    pass.pClearValues = &clear;
    vkCmdBeginRenderPass(command_, &pass, VK_SUBPASS_CONTENTS_INLINE);
    const float side = float(std::min(extent_.width, extent_.height));
    VkViewport viewport = {
        (extent_.width - side) / 2, (extent_.height - side) / 2, side, side, 0, 1};
    VkRect2D scissor = {{0, 0}, extent_};
    vkCmdSetViewport(command_, 0, 1, &viewport);
    vkCmdSetScissor(command_, 0, 1, &scissor);
    vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(
        command_, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &set_, 0, nullptr);
    vkCmdPushConstants(command_, layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &size_);
    vkCmdDraw(command_, 3, 1, 0, 0);
    vkCmdEndRenderPass(command_);
    ownership(false);
    check(vkEndCommandBuffer(command_));
    check(vkResetFences(device_, 1, &fence_));
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &acquired_;
    submit.pWaitDstStageMask = &stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &presented_[index];
    check(vkQueueSubmit(queue_, 1, &submit, fence_));
    VkPresentInfoKHR info = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &presented_[index];
    info.swapchainCount = 1;
    info.pSwapchains = &swap_;
    info.pImageIndices = &index;
    result = vkQueuePresentKHR(queue_, &info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
      resized_ = true;
    else
      check(result);
    return true;
  }
  std::vector<unsigned char> capture();
};
inline std::vector<unsigned char> ViewerPresenter::capture()
{
  std::lock_guard<std::mutex> lock(interop_mutex);
  wait_for_reader();
  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  const VkDeviceSize bytes = VkDeviceSize(size_) * size_ * 8;
  try {
    VkBufferCreateInfo buffer = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer.size = bytes;
    buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    check(vkCreateBuffer(device_, &buffer, nullptr, &staging));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, staging, &req);
    VkMemoryAllocateInfo allocate = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = req.size;
    allocate.memoryTypeIndex = memory_type(req.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkAllocateMemory(device_, &allocate, nullptr, &memory));
    check(vkBindBufferMemory(device_, staging, memory, 0));
    begin();
    ownership(true, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferCopy copy = {0, 0, bytes};
    vkCmdCopyBuffer(command_, buffer_, staging, 1, &copy);
    VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = staging;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(command_,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0,
                         0,
                         nullptr,
                         1,
                         &barrier,
                         0,
                         nullptr);
    ownership(false, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    check(vkEndCommandBuffer(command_));
    check(vkResetFences(device_, 1, &fence_));
    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_;
    check(vkQueueSubmit(queue_, 1, &submit, fence_));
    wait_for_reader();
    void *data;
    check(vkMapMemory(device_, memory, 0, bytes, 0, &data));
    const auto *half = static_cast<const uint16_t *>(data);
    std::vector<unsigned char> pixels(size_t(size_) * size_ * 4);
    auto decode = [](uint16_t h) {
      const unsigned exponent = (h >> 10) & 31, fraction = h & 1023;
      const float value = exponent == 0 ?
                              std::ldexp(float(fraction), -24) :
                              (exponent == 31 ?
                                   0.0f :
                                   std::ldexp(1.0f + float(fraction) / 1024, int(exponent) - 15));
      return (h & 32768) ? -value : value;
    };
    for (uint32_t y = 0; y < size_; ++y)
      for (uint32_t x = 0; x < size_; ++x) {
        const size_t src = (size_t(size_ - 1 - y) * size_ + x) * 4,
                     dst = (size_t(y) * size_ + x) * 4;
        for (int c = 0; c < 3; ++c) {
          float v = std::max(0.0f, decode(half[src + c]));
          v = v / (1 + v);
          v = v <= .0031308f ? 12.92f * v : 1.055f * std::pow(v, 1 / 2.4f) - .055f;
          pixels[dst + 2 - c] = static_cast<unsigned char>(std::clamp(v, 0.0f, 1.0f) * 255);
        }
        pixels[dst + 3] = 255;
      }
    vkUnmapMemory(device_, memory);
    vkDestroyBuffer(device_, staging, nullptr);
    vkFreeMemory(device_, memory, nullptr);
    return pixels;
  }
  catch (...) {
    vkDeviceWaitIdle(device_);
    if (staging)
      vkDestroyBuffer(device_, staging, nullptr);
    if (memory)
      vkFreeMemory(device_, memory, nullptr);
    throw;
  }
}

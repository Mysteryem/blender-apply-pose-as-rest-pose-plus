# Apply Pose as Rest Pose Plus Blender Add-on
Apply a pose as the rest pose *and* apply the pose to the meshes rigged to the armature. Supports meshes with shape keys!

Available in Pose mode from the `Pose` > `Apply` menu.

Supports Blender 3.3 and newer. The add-on may work with older versions of Blender, but no support will be provided for those versions.

![image](https://github.com/Mysteryem/blender-apply-pose-as-rest-pose-plus/assets/495015/8039d712-e3fa-44f5-97ed-1fbcce77532a)

Originally made for the Cats Blender Plugin to replace its slower implementation at the time, and to add support for bone rotation and non-uniform scaling of meshes with shape keys. Now separated into its own Add-on and made even faster with extra options.

New options:
- Apply the pose to only the selected bones.
- Override the Preserve Volume setting of each mesh's armature modifier.
- Ignore meshes with disabled armature modifiers.
- Choosing which rigged meshes to operate on:
  - The current view layer
  - The current scene
  - All meshes in the .blend
  - Compatibility mode for Cats that uses only meshes in the current view layer that are also parented to the armature (or have a parent, recursively, that is parented to the armature).

Performance changes:
- Much faster when the pose only changes the location of bones.
- Faster when the new pose only affects parts of meshes that are not affected by shape keys.
- Faster when there are lots of shape keys that do nothing.

Other:
- Error reporting when a rigged mesh cannot be modified (is multi-user data/is linked from a library/is stuck in Edit mode).

## Modes

### Fast

Fast mode assumes that the only vertices affected by the new pose are the vertices which are moved when applying the new pose to the Basis shape key.

In rare cases, this can skip vertices if those vertices are affected by multiple posed bones, but where the total effect of the posed bones cancels out. If the posed bones were rotated or scaled, on shape keys other than the Basis, the total effect of the posed bones might not cancel out.

### Exact

Exact mode checks through the vertices of each mesh in advance and finds all vertices that are rigged to the bones that have been posed or their child bones recursively.

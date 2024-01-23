bl_info = {
    "name": "Apply Pose as Rest Pose Plus",
    "author": "Mysteryem",
    "version": (0, 0, 1),
    "blender": (3, 3, 0),
    "location": "Pose > Apply > Apply Pose as Rest Pose Plus",
    "description": "Apply Pose as Rest Pose, but also applies to meshes rigged to the armature. Supports shape keys.",
    "doc_url": "https://github.com/Mysteryem/blender-apply-pose-as-rest-pose-plus",
    "tracker_url": "https://github.com/Mysteryem/blender-apply-pose-as-rest-pose-plus/issues",
    "category": "Rigging",
}
#     Apply Pose as Rest Pose Plus Blender add-on
#     Copyright (C) 2022-2024 Thomas Barlow (Mysteryem)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import bpy
from bpy.props import EnumProperty, BoolProperty
from bpy.types import (
    Armature,
    ArmatureModifier,
    Context,
    Depsgraph,
    Menu,
    Mesh,
    Object,
    Operator,
    Pose,
    PoseBone,
    ShapeKey,
)

from collections.abc import Iterable
from functools import cache
import numpy as np
from time import perf_counter
from types import SimpleNamespace
from typing import cast, TypeVar


# Utilities


T = TypeVar("T", bound=Operator)


def set_operator_description_from_doc(cls: type[T]) -> type[T]:
    """
    Clean up an Operator's .__doc__ for use as bl_description.
    :param cls: Operator subclass.
    :return: The Operator subclass argument.
    """
    doc = cls.__doc__
    if not doc or getattr(cls, "bl_description", None) is not None:
        # There is nothing to do if there is no `__doc__` or if `bl_description` has been set manually.
        return cls
    # Strip the entire string first to remove leading/trailing newlines and other whitespace.
    doc = doc.strip()
    if doc[-1] == ".":
        # Remove any trailing "." because Blender adds one automatically.
        doc = doc[:-1]
    # Remove leading/trailing whitespace from each line.
    cls.bl_description = "\n".join(line.strip() for line in doc.splitlines())
    return cls


def is_mesh_object_rigged_to_armature_object(armature_object: Object,
                                             armature_deform_bone_names: set[str],
                                             mesh_object: Object,
                                             ignore_disabled_modifiers: bool):
    """
    Determine if a mesh Object is rigged to an armature Object.
    :param armature_object:
    :param armature_deform_bone_names:
    :param mesh_object:
    :param ignore_disabled_modifiers:
    :return:
    """
    for mod in mesh_object.modifiers:
        if mod.type != 'ARMATURE':
            continue

        if ignore_disabled_modifiers and not mod.show_viewport:
            continue

        armature_mod = cast(ArmatureModifier, mod)

        if not armature_mod.use_vertex_groups or armature_mod.object != armature_object:
            continue

        for vertex_group in mesh_object.vertex_groups:
            if vertex_group.name in armature_deform_bone_names:
                return True

    return False


def get_objects_rigged_to_armature(armature_object: Object,
                                   objects_subset: Iterable[Object],
                                   ignore_disabled_modifiers: bool):
    assert armature_object.type == 'ARMATURE'
    armature_data = cast(Armature, armature_object.data)
    deform_bone_names = {bone.name for bone in armature_data.bones if bone.use_deform}

    rigged_objects = []
    for obj in objects_subset:
        if obj.type != 'MESH':
            continue
        if is_mesh_object_rigged_to_armature_object(armature_object, deform_bone_names, obj, ignore_disabled_modifiers):
            rigged_objects.append(obj)

    return rigged_objects


def get_directly_posed_bone_names(armature_pose: Pose):
    matrix_basis_identity = np.identity(4, dtype=np.single).reshape(1, 4, 4)

    bones = armature_pose.bones

    basis_matrices = np.empty((len(bones), 4, 4), dtype=np.single)
    basis_matrices_flat_view = basis_matrices.view()
    basis_matrices_flat_view.shape = -1
    # Note: The individual matrices are transposed compared to how they would normally be accessed through the Python
    # API. E.g. basis_matrices[0].T[1] is the same as bones[0].matrix_basis[1], giving the second row.
    bones.foreach_get("matrix_basis", basis_matrices_flat_view)
    is_close_to_identity = np.isclose(basis_matrices, matrix_basis_identity, rtol=1e-05, atol=1e-06)

    posed_bone_indices = np.flatnonzero(~np.all(is_close_to_identity, axis=(1, 2))).data

    if len(posed_bone_indices) != 0:
        is_close_no_translation = is_close_to_identity[:, :3]
        # The last value in the last row/column is most likely always going to be 1, but check it too.
        is_close_last_value = is_close_to_identity.ravel()[15::16]
        if np.all(is_close_no_translation) and np.all(is_close_last_value):
            translation_only = True
        else:
            translation_only = False
    else:
        translation_only = True

    return {bones[i].name for i in posed_bone_indices}, translation_only


def get_posed_deform_bone_names(armature_pose: Pose):
    """
    Get a set of posed bone names. This includes bones that have a posed parent, recursively.
    :param armature_pose:
    :return:
    """
    directly_posed_bone_names, translation_only = get_directly_posed_bone_names(armature_pose)

    @cache
    def is_pose_bone_posed(pose_bone: PoseBone):
        if pose_bone.name in directly_posed_bone_names:
            # Bone is posed directly.
            return True
        # Check bone's parent, recursively, is posed.
        parent = pose_bone.parent
        return parent is not None and is_pose_bone_posed(parent)

    return [pose_bone.name for pose_bone in armature_pose.bones
            if pose_bone.bone.use_deform and is_pose_bone_posed(pose_bone)], translation_only


def get_rigged_vertex_indices(mesh_object: Object, bone_names: Iterable[str]):
    vertex_group_index_lookup = {vertex_group.name: i for i, vertex_group in enumerate(mesh_object.vertex_groups)}
    vertex_group_indices = set()
    for bone_name in bone_names:
        idx = vertex_group_index_lookup.get(bone_name, -1)
        if idx != -1:
            vertex_group_indices.add(idx)

    mesh = cast(Mesh, mesh_object.data)

    if not vertex_group_indices:
        return np.empty(0, dtype=np.intp)

    def rigged_vertex_index_gen():
        for i, v in enumerate(mesh.vertices):
            for g in v.groups:
                if g.weight != 0 and g.group in vertex_group_indices:
                    yield i
                    break

    return np.fromiter(rigged_vertex_index_gen(), dtype=np.intp)


def _shape_key_co_memory_as_ndarray(shape_key: ShapeKey, do_check: bool = True):
    """
    ShapeKey.data elements have a dynamic typing based on the Object the shape keys are attached to, which makes them
    slower to access with foreach_get. For 'MESH' type Objects, the elements are always ShapeKeyPoint type and their
    data is stored contiguously, meaning an array can be constructed from the pointer to the first ShapeKeyPoint
    element.

    Creating an array from a pointer is inherently unsafe, so this function does a number of checks to make it safer.
    """
    if do_check and not _fast_mesh_shape_key_co_check():
        return None
    shape_data = shape_key.data
    num_co = len(shape_data)

    if num_co < 2:
        # At least 2 elements are required to check memory size.
        return None

    co_dtype = np.dtype(np.single)

    start_address = shape_data[0].as_pointer()
    last_element_start_address = shape_data[-1].as_pointer()

    memory_length_minus_one_item = last_element_start_address - start_address

    expected_element_size = co_dtype.itemsize * 3
    expected_memory_length = (num_co - 1) * expected_element_size

    if memory_length_minus_one_item == expected_memory_length:
        # Use NumPy's array interface protocol to construct an array from the pointer.
        array_interface_holder = SimpleNamespace(
            __array_interface__=dict(
                shape=(num_co * 3,),
                typestr=co_dtype.str,
                data=(start_address, False),  # False for writable
                version=3,
            )
        )
        return np.asarray(array_interface_holder)
    else:
        return None


# Initially set to None
_USE_FAST_SHAPE_KEY_CO_FOREACH_GETSET = None


def _fast_mesh_shape_key_co_check():
    global _USE_FAST_SHAPE_KEY_CO_FOREACH_GETSET
    if _USE_FAST_SHAPE_KEY_CO_FOREACH_GETSET is not None:
        # The check has already been run and the result has been stored in _USE_FAST_SHAPE_KEY_CO_FOREACH_GETSET.
        return _USE_FAST_SHAPE_KEY_CO_FOREACH_GETSET

    tmp_mesh = None
    tmp_object = None
    try:
        tmp_mesh = bpy.data.meshes.new("")
        num_co = 100
        tmp_mesh.vertices.add(num_co)
        # An Object is needed to add/remove shape keys from a Mesh.
        tmp_object = bpy.data.objects.new("", tmp_mesh)
        shape_key = tmp_object.shape_key_add(name="")
        shape_data = shape_key.data

        if shape_key.bl_rna.properties["data"].fixed_type == shape_data[0].bl_rna:
            # The shape key "data" collection is no longer dynamically typed and foreach_get/set should be fast enough.
            _USE_FAST_SHAPE_KEY_CO_FOREACH_GETSET = False
            return False

        co_dtype = np.dtype(np.single)

        # Fill the shape key with some data.
        shape_data.foreach_set("co", np.arange(3 * num_co, dtype=co_dtype))

        # The check is this function, so explicitly don't do the check.
        co_memory_as_array = _shape_key_co_memory_as_ndarray(shape_key, do_check=False)
        if co_memory_as_array is not None:
            # Immediately make a copy in case the `foreach_get` afterward can cause the memory to be reallocated.
            co_array_from_memory = co_memory_as_array.copy()
            del co_memory_as_array
            # Check that the array created from the pointer has the exact same contents as using foreach_get.
            co_array_check = np.empty(num_co * 3, dtype=co_dtype)
            shape_data.foreach_get("co", co_array_check)
            if np.array_equal(co_array_check, co_array_from_memory, equal_nan=True):
                _USE_FAST_SHAPE_KEY_CO_FOREACH_GETSET = True
                return True

        # Something didn't work.
        print("Fast shape key co access failed. Access will fall back to regular foreach_get/set.")
        _USE_FAST_SHAPE_KEY_CO_FOREACH_GETSET = False
        return False
    finally:
        # Clean up temporary objects.
        if tmp_object is not None:
            tmp_object.shape_key_clear()
            bpy.data.objects.remove(tmp_object)
        if tmp_mesh is not None:
            bpy.data.meshes.remove(tmp_mesh)


# ShapeKey.points was added in Blender 4.1.0.
_HAS_SHAPE_KEY_POINTS = "points" in bpy.types.ShapeKey.bl_rna.properties and bpy.app.version >= (4, 1, 0)

if _HAS_SHAPE_KEY_POINTS:
    def fast_mesh_shape_key_co_foreach_get(shape_key: ShapeKey, arr: np.ndarray):
        shape_key.points.foreach_get("co", arr)

    def fast_mesh_shape_key_co_foreach_set(shape_key: ShapeKey, arr: np.ndarray):
        shape_key.points.foreach_set("co", arr)
else:
    def fast_mesh_shape_key_co_foreach_get(shape_key: ShapeKey, arr: np.ndarray):
        co_memory_as_array = _shape_key_co_memory_as_ndarray(shape_key)
        if co_memory_as_array is not None:
            arr[:] = co_memory_as_array
        else:
            shape_key.data.foreach_get("co", arr)

    def fast_mesh_shape_key_co_foreach_set(shape_key: ShapeKey, arr: np.ndarray):
        co_memory_as_array = _shape_key_co_memory_as_ndarray(shape_key)
        if co_memory_as_array is not None:
            co_memory_as_array[:] = arr
            # Unsure if this is required. I don't think `.update()` actually does anything, nor do I think #foreach_set
            # calls any function equivalent to `.update()` either.
            # Memory has been set directly, so call `.update()`.
            shape_key.data.update()
        else:
            shape_key.data.foreach_set("co", arr)


# Blender Classes


@set_operator_description_from_doc
class ApplyPoseAsRestPosePlus(Operator):
    """
    Apply the current pose as the new rest pose *and* apply the pose to meshes rigged to the armature.
    """
    bl_idname = "mysteryem.apply_pose_as_rest_pose_plus"
    bl_label = "Apply Pose as Rest Pose Plus"
    bl_options = {'REGISTER', 'UNDO'}

    preserve_volume: EnumProperty(
        items=[
            ('MODIFIER', "Use Modifier Settings", "Use the Preserve Volume setting of each Armature modifier when"
                                                  " applying"),
            ('ENABLED', "Enabled", "Enable Preserve Volume when applying"),
            ('DISABLED', "Disabled", "Disable Preserve Volume when applying"),
        ],
        name="Preserve Volume",
        description="Use the Preserve Volume setting when applying the pose to meshes",
        default='MODIFIER',
    )

    mesh_objects_subset: EnumProperty(
        items=[
            ('VIEW_LAYER', "View Layer", "Mesh objects in the current view layer"),
            ('SCENE', "Scene", "Mesh objects in the current scene"),
            ('ALL', "All", "All mesh objects"),
            ('CATS', "Cats (View Layer and parented)", "Mesh objects in the current view layer that are parented to the"
                                                       " armature or one of its children. This almost exactly matches"
                                                       " the behaviour of Cats's \"Apply as Rest Pose\""),
        ],
        name="Meshes Subset",
        description="The subset of mesh objects, that are rigged to the armature, to apply the pose to",
        default='VIEW_LAYER',
    )

    ignore_disabled_modifiers: BoolProperty(
        name="Ignore disabled modifiers",
        description="Ignore disabled armature modifiers when checking if a mesh is rigged to the armature",
        default=False,
    )

    performance_mode: EnumProperty(
        items=[
            ('FAST', "Fast", "Only the vertices of the Basis that end up in a new position from the pose are checked."
                             " This can skip vertices that are moved by multiple bones but end up in the same place"),
            ('EXACT', "Exact", "All vertices that are rigged to the armature are checked"),
            ('DEBUG', "Debug", "Assumes all vertices are affected by the new pose"),
        ],
        name="Mode",
        default='FAST',
    )

    selected: BoolProperty(
        name="Selected",
        description="Only affect selected bones",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        pose_object = context.pose_object
        if not pose_object or not pose_object.mode == 'POSE' or not pose_object.type == 'ARMATURE':
            cls.poll_message_set("An armature in pose mode is required.")
            return False
        return True

    def validate_objects(self, objects: Iterable[Object]):
        for obj in objects:
            if obj.data.users > 1:
                # When the mesh has no shape keys, modifiers are applied, which cannot be applied to multi-user meshes.
                # When the mesh has shape keys, the shape keys are adjusted manually, but if there are two mesh Objects
                # with the same data that are rigged to the armature, then the shape keys would be adjusted twice,
                # almost assuredly ending with incorrect results.
                self.report({'ERROR'}, f"Cannot be applied to multi-user meshes: {obj!r}")
                return False
            if obj.library is not None:
                self.report({'ERROR'}, f"Cannot be applied to meshes linked from a library: {obj!r}")
                return False
            if obj.mode == 'EDIT':
                self.report({'ERROR'}, f"Cannot be applied to meshes that are in Edit mode: {obj!r}")
                return False
        return True

    def execute(self, context: Context) -> set[str]:
        # `poll` guarantees it exists and is an armature Object.
        armature_obj = context.pose_object

        match self.mesh_objects_subset:
            case 'VIEW_LAYER':
                objects = context.view_layer.objects
            case 'SCENE':
                objects = context.scene.objects
            case 'ALL':
                objects = bpy.data.objects
            case 'CATS':
                objects = set(context.view_layer.objects)
                objects.intersection_update(armature_obj.children_recursive)
            case _:
                self.report({'ERROR'}, "Unexpected subset '%s'" % self.mesh_objects_subset)
                return {'CANCELLED'}

        match self.preserve_volume:
            case 'ENABLED':
                preserve_volume_override = True
            case 'DISABLED':
                preserve_volume_override = False
            case 'MODIFIER':
                preserve_volume_override = None
            case _:
                self.report({'ERROR'}, "Unexpected Preserve Volume setting '%s'" % self.preserve_volume)
                return {'CANCELLED'}

        rigged_mesh_objects = get_objects_rigged_to_armature(armature_obj, objects, self.ignore_disabled_modifiers)

        # Ensure that the Operator can be applied to all the Objects.
        if not self.validate_objects(rigged_mesh_objects):
            return {'CANCELLED'}

        deselected_bone_indices = None
        deselected_bone_rotations = None
        deselected_bone_scales = None
        deselected_bone_locations = None
        try:
            if self.selected:
                # Temporarily clear the pose of deselected bones, apply the armature as normal and then restore the pose
                # of the deselected bones.
                pose_bones = armature_obj.pose.bones
                bones = cast(Armature, armature_obj.data).bones
                selected_mask = np.empty(len(bones), dtype=bool)
                hide_mask = np.empty(len(bones), dtype=bool)
                bones.foreach_get("select", selected_mask)
                bones.foreach_get("hide", hide_mask)
                # Consider hidden bones to always be deselected.
                deselected_bone_indices = np.flatnonzero((~selected_mask) | hide_mask)

                # Get current locations.
                locations = np.empty((len(pose_bones), 3), dtype=np.single)
                locations_flat = locations.view()
                locations_flat.shape = -1
                pose_bones.foreach_get("location", locations_flat)
                deselected_bone_locations = locations[deselected_bone_indices]
                # Set the location of all deselected bones to (0, 0, 0).
                locations[deselected_bone_indices] = np.zeros((1, 3), dtype=locations.dtype)
                pose_bones.foreach_set("location", locations_flat)

                # Get current scales.
                scales = np.empty((len(pose_bones), 3), dtype=np.single)
                scales_flat = scales.view()
                scales_flat.shape = -1
                pose_bones.foreach_get("scale", scales_flat)
                deselected_bone_scales = scales[deselected_bone_indices]
                # Set the scale of all deselected bones to (1, 1, 1).
                scales[deselected_bone_indices] = np.ones((1, 3), dtype=scales.dtype)
                pose_bones.foreach_set("scale", scales_flat)

                # Get current rotations.
                # Each bone can have a different rotation mode, so foreach_get/foreach_set cannot be used.
                deselected_bone_rotations = []
                axis_angle_identity = (0.0, 0.0, 0.0, 0.0)
                for i in deselected_bone_indices.data:
                    pose_bone = pose_bones[i]
                    match pose_bone.rotation_mode:
                        case 'QUATERNION':
                            pose_bone_rotation = pose_bone.rotation_quaternion
                            rotation = pose_bone_rotation.copy()
                            # Set to identity.
                            pose_bone_rotation.identity()
                        case 'AXIS_ANGLE':
                            rotation = tuple(pose_bone.rotation_axis_angle)
                            # Set to identity.
                            pose_bone.rotation_axis_angle = axis_angle_identity
                        case _:
                            pose_bone_rotation = pose_bone.rotation_euler
                            rotation = pose_bone_rotation.copy()
                            # Set to identity.
                            pose_bone_rotation.zero()
                    deselected_bone_rotations.append(rotation)

                # Ensure the changes to pose bone transforms are taken into account.
                armature_obj.update_tag(refresh={'OBJECT'})

            posed_deform_bone_names, translation_only = get_posed_deform_bone_names(armature_obj.pose)

            if len(posed_deform_bone_names) != 0:
                mesh_processing_start = perf_counter()
                for mesh_obj in rigged_mesh_objects:
                    mesh = cast(Mesh, mesh_obj.data)
                    if mesh.shape_keys and mesh.shape_keys.key_blocks:
                        # The mesh has shape keys
                        shape_keys = mesh.shape_keys
                        key_blocks = shape_keys.key_blocks
                        if len(key_blocks) == 1:
                            # The mesh only has a basis shape key, so it can be removed it and added back afterward.
                            # Get Reference Key.
                            reference_shape_key = key_blocks[0]
                            # Save the name of the Reference Key.
                            original_basis_name = reference_shape_key.name
                            # Remove the basis shape key so there are now no shape keys.
                            mesh_obj.shape_key_remove(reference_shape_key)
                            # Apply the pose to the mesh.
                            apply_armature_to_mesh_with_no_shape_keys(context,
                                                                      armature_obj,
                                                                      mesh_obj,
                                                                      preserve_volume_override)
                            # Add the basis shape key back with the same name as before.
                            mesh_obj.shape_key_add(name=original_basis_name)
                        else:
                            # Apply the pose to the mesh, taking into account the shape keys.
                            if translation_only:
                                # Optimised for a new pose which only translates bones.
                                apply_armature_to_mesh_with_shape_keys_translation_only(context,
                                                                                        armature_obj,
                                                                                        mesh_obj,
                                                                                        preserve_volume_override)
                            else:
                                apply_armature_to_mesh_with_shape_keys(context,
                                                                       armature_obj,
                                                                       mesh_obj,
                                                                       posed_deform_bone_names,
                                                                       preserve_volume_override,
                                                                       self.performance_mode)
                    else:
                        # The mesh doesn't have shape keys, so we can easily apply the pose to the mesh.
                        apply_armature_to_mesh_with_no_shape_keys(context, armature_obj, mesh_obj, preserve_volume_override)

                mesh_processing_end = perf_counter()
                print(f"Mesh processing took {(mesh_processing_end - mesh_processing_start) * 1000:f}ms")
            else:
                print("No deform bones (or their recursive parents) were posed")

            # Once the mesh and shape keys (if any) have been applied, the last step is to apply the current pose of the
            # bones as the new rest pose.
            #
            # From the poll function, armature_obj must be in pose mode and be the `.pose_object`, but maybe it's possible
            # it might not be the active object. We can use an operator override to tell the operator to treat armature_obj
            # as if it's the active object even if it's not, skipping the need to actually set armature_obj as the active
            # object.
            with context.temp_override(active_object=armature_obj):
                bpy.ops.pose.armature_apply(selected=self.selected)
        finally:
            # If the pose was only applied to selected bones, restore the matrices of deselected bones.
            if self.selected and deselected_bone_indices is not None:
                pose_bones = armature_obj.pose.bones
                if deselected_bone_locations is not None:
                    locations = np.empty((len(pose_bones), 3), dtype=np.single)
                    locations_flat = locations.view()
                    locations_flat.shape = -1
                    pose_bones.foreach_get("location", locations_flat)
                    locations[deselected_bone_indices] = deselected_bone_locations
                    pose_bones.foreach_set("location", locations_flat)

                if deselected_bone_scales is not None:
                    scales = np.empty((len(pose_bones), 3), dtype=np.single)
                    scales_flat = scales.view()
                    scales_flat.shape = -1
                    pose_bones.foreach_get("scale", scales_flat)
                    scales[deselected_bone_indices] = deselected_bone_scales
                    pose_bones.foreach_set("scale", scales_flat)

                if deselected_bone_rotations is not None:
                    for i, rotation in zip(deselected_bone_indices.data, deselected_bone_rotations):
                        pose_bone = pose_bones[i]
                        match pose_bone.rotation_mode:
                            case 'QUATERNION':
                                pose_bone.rotation_quaternion = rotation
                            case 'AXIS_ANGLE':
                                pose_bone.rotation_axis_angle = rotation
                            case _:
                                pose_bone.rotation_euler = rotation

                # Ensure the pose bones are visibly updated.
                armature_obj.update_tag(refresh={'OBJECT'})

        self.report({'INFO'}, "Pose successfully applied as rest pose.")
        return {'FINISHED'}


# Implementation


def _apply_armature_modifier_to_mesh(context: Context,
                                     armature_obj: Object,
                                     mesh_obj: Object,
                                     preserve_volume_override: bool | None,
                                     as_shape_key: bool):
    armature_mod = cast(ArmatureModifier, mesh_obj.modifiers.new("PoseToRest", 'ARMATURE'))
    armature_mod.object = armature_obj
    if preserve_volume_override is not None:
        armature_mod.use_deform_preserve_volume = preserve_volume_override
    # In the unlikely case that there was already a modifier with the same name as the new modifier, the new modifier
    # will have ended up with a different name.
    mod_name = armature_mod.name
    # Context override to let us run the modifier operators on mesh_obj, even if it's not the active object.
    # Moving the modifier to the first index will prevent an Info message about the applied modifier not being first and
    # potentially having unexpected results.
    with context.temp_override(object=mesh_obj):
        if bpy.app.version >= (3, 5):
            # Blender 3.5 adds a nice method for reordering modifiers.
            mesh_obj.modifiers.move(mesh_obj.modifiers.find(mod_name), 0)
        else:
            bpy.ops.object.modifier_move_to_index(modifier=mod_name, index=0)
        if as_shape_key:
            # Since Blender 3.0, the effect of shape keys are included in the saved shape key...
            # https://projects.blender.org/blender/blender/issues/91923
            mesh = cast(Mesh, mesh_obj.data)
            shape_keys = mesh.shape_keys
            if shape_keys:
                orig_active_shape_index = mesh_obj.active_shape_key_index
                orig_show_only_shape = mesh_obj.show_only_shape_key
                orig_reference_key_mute = shape_keys.reference_key.mute
                try:
                    # Currently, the reference shape key is always the first shape key.
                    mesh_obj.active_shape_key_index = 0
                    mesh_obj.show_only_shape_key = True
                    shape_keys.reference_key.mute = False
                    bpy.ops.object.modifier_apply_as_shapekey(modifier=mod_name)
                finally:
                    mesh_obj.active_shape_key_index = orig_active_shape_index
                    mesh_obj.show_only_shape_key = orig_show_only_shape
                    shape_keys.reference_key.mute = orig_reference_key_mute
            else:
                bpy.ops.object.modifier_apply_as_shapekey(modifier=mod_name)
        else:
            bpy.ops.object.modifier_apply(modifier=mod_name)


def apply_armature_to_mesh_with_no_shape_keys(context: Context,
                                              armature_obj: Object,
                                              mesh_obj: Object,
                                              preserve_volume_override: bool | None):
    _apply_armature_modifier_to_mesh(context, armature_obj, mesh_obj, preserve_volume_override, False)


def apply_armature_to_mesh_with_shape_keys_translation_only(context: Context,
                                                            armature_obj: Object,
                                                            mesh_obj: Object,
                                                            preserve_volume_override: bool | None):
    # When a new pose is only translation, the effect on all shape keys will be the same, so the Armature modifier can
    # be applied as a shape key and then that shape key can be applied to all other shape keys.
    _apply_armature_modifier_to_mesh(context, armature_obj, mesh_obj, preserve_volume_override, True)
    # The newly added shape key will be at the bottom.
    mesh = cast(Mesh, mesh_obj.data)
    shape_keys = mesh.shape_keys
    key_blocks = shape_keys.key_blocks
    new_shape_key = key_blocks[-1]
    new_shape_key_relative = new_shape_key.relative_key
    num_co = len(mesh.vertices) * 3
    new_key_cos = np.empty(num_co, dtype=np.single)
    new_key_relative_cos = np.empty(num_co, dtype=np.single)
    fast_mesh_shape_key_co_foreach_get(new_shape_key, new_key_cos)
    fast_mesh_shape_key_co_foreach_get(new_shape_key_relative, new_key_relative_cos)
    difference = new_key_cos - new_key_relative_cos
    # The same as the default argument.
    rtol = 1e-05
    # The default argument is 1e-08, but Blender shape key coordinates and mesh positions are single-precision float, so
    # increase the absolute tolerance slightly.
    atol = 1e-06
    if np.allclose(difference, 0, rtol=rtol, atol=atol, equal_nan=True):
        print(f"Skipped '{mesh_obj.name}' because it is not affected by the new pose")
    else:
        # `new_key_relative_cos` isn't needed any more, so re-use it to store other shape key cos.
        shape_key_cos = new_key_relative_cos
        # Array to store updated shape keys in, to avoid allocating a new array each time.
        shape_key_cos_updated = np.empty_like(shape_key_cos)
        reference_key = shape_keys.reference_key
        # Apply the newly added shape key to every other shape key.
        for shape_key in key_blocks[:-1]:
            # Get the new coordinates for the shape key.
            if shape_key == new_shape_key_relative:
                # The result of applying the new shape key to its relative key is simply itself.
                updated_shape_key_cos = new_key_cos
            else:
                fast_mesh_shape_key_co_foreach_get(shape_key, shape_key_cos)
                updated_shape_key_cos = np.add(shape_key_cos, difference, out=shape_key_cos_updated)

            # Update the shape key.
            fast_mesh_shape_key_co_foreach_set(shape_key, updated_shape_key_cos)

            if shape_key == reference_key:
                # Mesh positions must also be updated to match the reference shape key.
                if bpy.app.version >= (3, 5):
                    position_attribute = mesh.attributes.get("position")
                    if (position_attribute
                            and position_attribute.data_type == 'FLOAT_VECTOR'
                            and position_attribute.domain == 'POINT'):
                        position_attribute.data.foreach_get("vector", updated_shape_key_cos)
                else:
                    mesh.vertices.foreach_get("co", updated_shape_key_cos)
    # Remove the newly added shape key.
    mesh_obj.shape_key_remove(new_shape_key)


def _apply_armature_to_mesh_with_shape_keys_impl(context: Context,
                                                 mesh_obj: Object,
                                                 me: Mesh,
                                                 performance_mode: str,
                                                 all_vertices_affected: bool,
                                                 affected_rigged_vertex_indices):
    # Coordinates are xyz positions and get flattened when using the foreach_set/foreach_get functions, so the array
    # length will be 3 times the number of vertices.
    co_length = len(me.vertices) * 3
    # We can re-use the same array over and over
    eval_verts_cos_array = np.empty(co_length, dtype=np.single)

    # A Depsgraph lets us evaluate objects and get their state after the effect of modifiers and shape keys.
    depsgraph: Depsgraph | None = None
    evaluated_mesh_obj = None

    has_position_attribute = bpy.app.version >= (3, 5)

    def get_eval_cos_array():
        nonlocal depsgraph
        nonlocal evaluated_mesh_obj
        # Get the Depsgraph and evaluate the mesh if we haven't done so already.
        if depsgraph is None or evaluated_mesh_obj is None:
            depsgraph = context.evaluated_depsgraph_get()
            evaluated_mesh_obj = mesh_obj.evaluated_get(depsgraph)
        else:
            # If we already have the depsgraph and evaluated mesh, in order for the change to the active shape key to
            # take effect, the depsgraph has to be updated.
            depsgraph.update()
        # Get the cos of the vertices from the evaluated mesh.
        evaluated_mesh = cast(Mesh, evaluated_mesh_obj.data)
        if has_position_attribute:
            position_attribute = evaluated_mesh.attributes.get("position")
            if (position_attribute
                    and position_attribute.data_type == 'FLOAT_VECTOR'
                    and position_attribute.domain == 'POINT'):
                position_attribute.data.foreach_get("vector", eval_verts_cos_array)
        else:
            evaluated_mesh.vertices.foreach_get("co", eval_verts_cos_array)
        return eval_verts_cos_array

    # Same as default
    rtol = 1e-05
    # Default is 1e-08, but Blender shape key coordinates and mesh positions are single-precision float.
    atol = 1e-06

    print(f"Processing '{mesh_obj.name}'")

    key_blocks = me.shape_keys.key_blocks
    skip_key_blocks = set()
    for i, shape_key in enumerate(key_blocks):
        if i in skip_key_blocks:
            # The shape key could be processed without needing to evaluate the mesh, so continue to the next shape key.
            continue
        # As shape key pinning is enabled, when we change the active shape key, it will change the state of the mesh.
        mesh_obj.active_shape_key_index = i
        # The cos of the vertices of the evaluated mesh include the effect of the pinned shape key and all the
        # modifiers (in this case, only the armature modifier we added since all the other modifiers are disabled in the
        # viewport).
        # This combination gives the same effect as if we'd applied the armature modifier to a mesh with the same shape
        # as the active shape key, so we can simply set the shape key to the evaluated mesh position.
        #
        # Get the evaluated coordinates.
        evaluated_cos = get_eval_cos_array()
        if i == 0:
            # Find which shape keys are affected the same as the reference key and can therefore have those affected
            # areas set the same.
            # Consider an eye blink shape key when the new pose only rotates the arms. The affected area of the model
            # will only be the arms, so the affected area in the eye blink shape key can be set to the affected area of
            # the reference shape key with the new pose applied.
            original_reference_key_cos = np.empty_like(evaluated_cos)
            original_reference_key_cos_vector_view = original_reference_key_cos.view()
            original_reference_key_cos_vector_view.shape = (-1, 3)
            fast_mesh_shape_key_co_foreach_get(shape_key, original_reference_key_cos)
            evaluated_cos_vector_view = evaluated_cos.view()
            evaluated_cos_vector_view.shape = (-1, 3)
            if performance_mode == 'FAST':
                # Assume the affected vertices in the reference key are the only affected vertices of all other shape
                # keys too. This will almost always be the case unless multiple bones move a vertex, but their combined
                # effect cancels out. If some of those bones were rotated or scaled, then in other shape keys, the
                # combined effect might not cancel out.
                unaffected_by_pose_mask = np.isclose(evaluated_cos_vector_view,
                                                     original_reference_key_cos_vector_view,
                                                     rtol=rtol,
                                                     atol=atol,
                                                     equal_nan=True)
                unaffected_by_pose_vector_mask = np.all(unaffected_by_pose_mask, axis=1)
                # Invert in-place and assign a new variable.
                affected_by_pose_vector_mask = np.logical_not(unaffected_by_pose_vector_mask,
                                                              out=unaffected_by_pose_vector_mask)
                affected_rigged_vertex_indices = np.flatnonzero(affected_by_pose_vector_mask)
                if len(affected_rigged_vertex_indices) == 0:
                    # No vertices are affected by the new pose, so there is nothing to do.
                    print(f"Skipped '{mesh_obj.name}' because it is not affected by the new pose")
                    break
                all_vertices_affected = len(affected_rigged_vertex_indices) == len(me.vertices)
            if all_vertices_affected:
                # If the root bone, e.g. hips, is transformed, all vertices will be affected.
                affected_vertices_evaluated_vectors = evaluated_cos_vector_view
            else:
                affected_vertices_evaluated_vectors = evaluated_cos_vector_view[affected_rigged_vertex_indices]

            # Re-use the same array for each non-reference shape key iterated.
            original_other_key_cos = np.empty_like(evaluated_cos)
            original_other_key_cos_vector_view = original_other_key_cos.view()
            original_other_key_cos_vector_view.shape = (-1, 3)
            for j, sk in enumerate(key_blocks[1:], start=1):
                fast_mesh_shape_key_co_foreach_get(sk, original_other_key_cos)
                close_to_reference_key_mask = np.isclose(original_other_key_cos,
                                                         original_reference_key_cos,
                                                         rtol=rtol,
                                                         atol=atol,
                                                         equal_nan=True)
                close_to_reference_key_mask.shape = (-1, 3)
                if close_to_reference_key_mask.all():
                    print(f"\tshape '{key_blocks[j].name}' is the same as the reference key")
                    # Set the shape key to the same as the evaluated reference key.
                    fast_mesh_shape_key_co_foreach_set(sk, evaluated_cos)
                    skip_key_blocks.add(j)
                elif (not all_vertices_affected
                      and close_to_reference_key_mask.any()
                      and close_to_reference_key_mask[affected_rigged_vertex_indices].all()):
                    print(f"\tshape '{key_blocks[j].name}' is not affected by the new pose")
                    # The parts of the shape key that are affected by the pose are the same as the reference key.
                    # Set those parts to the same parts in the evaluated reference key.
                    original_other_key_cos_vector_view[affected_rigged_vertex_indices] = affected_vertices_evaluated_vectors
                    fast_mesh_shape_key_co_foreach_set(sk, original_other_key_cos)
                    skip_key_blocks.add(j)

        # And set the shape key to those same cos.
        fast_mesh_shape_key_co_foreach_set(shape_key, evaluated_cos)
        # If it's the reference shape key, we also have to set the mesh vertices to match, otherwise the two will be
        # desynced until Edit mode has been entered and exited, which can cause odd behaviour when creating shape keys
        # with from_mix=False or when removing all shape keys.
        if i == 0:
            mesh_data = cast(Mesh, me)
            if has_position_attribute:
                position_attribute = mesh_data.attributes.get("position")
                if (position_attribute
                        and position_attribute.data_type == 'FLOAT_VECTOR'
                        and position_attribute.domain == 'POINT'):
                    position_attribute.data.foreach_set("vector", evaluated_cos)
            else:
                mesh_data.vertices.foreach_set("co", evaluated_cos)


def apply_armature_to_mesh_with_shape_keys(context: Context,
                                           armature_obj: Object,
                                           mesh_obj: Object,
                                           posed_deform_bone_names: Iterable[str],
                                           preserve_volume_override: bool | None,
                                           performance_mode: str):
    me = cast(Mesh, mesh_obj.data)

    if performance_mode == 'EXACT':
        # Find the indices of all vertices that are rigged to a posed bone (or a bone with a posed parent recursively).
        affected_rigged_vertex_indices = get_rigged_vertex_indices(mesh_obj, posed_deform_bone_names)
        if len(affected_rigged_vertex_indices) == 0:
            # No vertices are affected by the new pose, so there is nothing to do.
            print(f"Skipped '{mesh_obj.name}' because it is not affected by the new pose")
            return
        all_vertices_affected = len(me.vertices) == len(affected_rigged_vertex_indices)
    else:
        # 'FAST' mode will change these in the implementation, otherwise it is assumed that all vertices are affected by
        # the new pose.
        affected_rigged_vertex_indices = None
        all_vertices_affected = True

    # Store the current values of properties that will be changed to apply the armature, so that the properties can be
    # restored to their original values afterwards.
    old_active_shape_key_index = mesh_obj.active_shape_key_index
    old_show_only_shape_key = mesh_obj.show_only_shape_key
    key_blocks = me.shape_keys.key_blocks
    shape_key_vertex_groups_and_mutes = [(sk.vertex_group, sk.mute) for sk in key_blocks]
    modifier_show_viewports = [mod.show_viewport for mod in mesh_obj.modifiers]

    armature_mod_name = None

    try:
        # Shape key pinning shows the active shape key in the viewport without blending; effectively what you see when
        # in edit mode. Combined with an armature modifier, we can use this to figure out the correct positions for all
        # the shape keys.
        mesh_obj.show_only_shape_key = True
        key_blocks = me.shape_keys.key_blocks
        # Remove vertex_groups from and disable mutes on shape keys because they affect pinned shape keys.
        for shape_key in key_blocks:
            shape_key.vertex_group = ''
            shape_key.mute = False
        # Disable all modifiers from showing in the viewport so that they have no effect.
        for mod in mesh_obj.modifiers:
            mod.show_viewport = False

        # Temporarily add a new armature modifier.
        armature_mod = cast(ArmatureModifier, mesh_obj.modifiers.new("PoseToRest", 'ARMATURE'))
        armature_mod_name = armature_mod.name
        armature_mod.object = armature_obj
        if preserve_volume_override is not None:
            armature_mod.use_deform_preserve_volume = preserve_volume_override

        _apply_armature_to_mesh_with_shape_keys_impl(context, mesh_obj, me, performance_mode, all_vertices_affected,
                                                     affected_rigged_vertex_indices)
    finally:
        # Remove the temporarily added armature modifier.
        if armature_mod_name is not None:
            mesh_obj.modifiers.remove(mesh_obj.modifiers[armature_mod_name])
        # Restore modifiers `.show_viewport` to their original values.
        for mod, orig_show_viewport in zip(mesh_obj.modifiers, modifier_show_viewports):
            mod.show_viewport = orig_show_viewport
        # Restore shape key vertex groups and mutes.
        for shape_key, (vertex_group, mute) in zip(me.shape_keys.key_blocks, shape_key_vertex_groups_and_mutes):
            shape_key.vertex_group = vertex_group
            shape_key.mute = mute
        # Restore active shape key index.
        mesh_obj.active_shape_key_index = old_active_shape_key_index
        # Restore `.show_only_shape_key`/'Shape key pinning'.
        mesh_obj.show_only_shape_key = old_show_only_shape_key


# Registration

def draw_in_menu(self: Menu, context: Context):
    self.layout.separator()
    self.layout.operator(ApplyPoseAsRestPosePlus.bl_idname).selected = False
    self.layout.operator(ApplyPoseAsRestPosePlus.bl_idname, text="Apply Selected as Rest Pose Plus").selected = True


# Links for Right Click -> Online Manual.
def add_manual_map():
    url_manual_prefix = "https://github.com/Mysteryem/blender-apply-pose-as-rest-pose-plus"
    url_manual_mapping = (
        ("bpy.ops." + ApplyPoseAsRestPosePlus.bl_idname, ""),
    )
    return url_manual_prefix, url_manual_mapping


def register():
    bpy.utils.register_class(ApplyPoseAsRestPosePlus)
    bpy.utils.register_manual_map(add_manual_map)
    bpy.types.VIEW3D_MT_pose_apply.append(draw_in_menu)


def unregister():
    bpy.types.VIEW3D_MT_pose_apply.remove(draw_in_menu)
    bpy.utils.unregister_manual_map(add_manual_map)
    bpy.utils.unregister_class(ApplyPoseAsRestPosePlus)


# For testing in Blender's Text Editor.
if __name__ == "__main__":
    # Try and unregister the previously registered version.
    unregister_attribute = "apply_pose_as_rest_pose_plus_unregister_old"
    temp_storage = bpy.types.WindowManager
    if old_unregister := getattr(temp_storage, unregister_attribute, None):
        try:
            old_unregister()
        except Exception as e:
            print(e)
    register()
    setattr(temp_storage, unregister_attribute, unregister)

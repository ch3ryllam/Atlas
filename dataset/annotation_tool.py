"""
Usage:
python dataset/annotation_tool.py -i banana
python dataset/annotation_tool.py -i banana --frames_dir custom/path/banana --json_dir custom/json_dir
"""

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import os
import glob
import sys
import os.path as osp

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))
from utils import rle_to_bmask, bmask_to_rle


class AnnotationToolApp:
    def __init__(self, root, frames_dir, json_dir, instance_name):
        self.root = root
        self.root.title(f"Annotation Tool - Instance: {instance_name}")

        self.frames_dir = frames_dir
        self.json_dir = json_dir
        self.instance_name = instance_name

        # Load Video Frames
        self.frames = sorted(glob.glob(os.path.join(self.frames_dir, "*.jpg")))
        if not self.frames:
            messagebox.showerror("Error", f"No frames found in {self.frames_dir}")
            sys.exit(1)

        sample_frame = cv2.imread(self.frames[0])
        self.orig_H, self.orig_W = sample_frame.shape[:2]

        # Application State
        self.frame_idx = 0
        self.mode = "ADD"
        self.poly_points = []
        self.mask_modified = False

        # 1-Step Undo State Buffer
        self.undo_state = None

        # Load CropFormer proposals (Lazy Loading for Memory Efficiency)
        json_path = osp.join(self.json_dir, f"{self.instance_name}.json")
        print(
            f">>> Loading CropFormer proposals for instance '{self.instance_name}'..."
        )
        self.proposals = self.load_cropformer_masks(json_path)

        # Multi-Object Tracking State
        self.objects = {}
        self.object_masks = {}
        self.bad_masks = {}  # Tracks boolean flag for bad masks per object per frame
        self.active_obj_id = None
        self.next_obj_id = 0

        # UI State for resizing
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0

        self.setup_ui()
        self.load_existing_annotations()
        self.update_view()

    def setup_ui(self):
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # Left Side: Canvas & Controls
        left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(left_frame, weight=3)

        control_frame = ttk.Frame(left_frame, padding="5")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(control_frame, text="Prev (A)", command=self.prev_frame).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(control_frame, text="Next (D)", command=self.next_frame).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(
            control_frame, text="Mark Bad & Next (F)", command=self.mark_bad_and_next
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Unmark Bad (U)", command=self.unmark_bad).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Label(control_frame, text="Jump to:").pack(side=tk.LEFT, padx=(10, 2))
        self.frame_jump_var = tk.StringVar(value="0")
        self.frame_entry = ttk.Entry(
            control_frame, textvariable=self.frame_jump_var, width=5
        )
        self.frame_entry.pack(side=tk.LEFT)
        self.frame_entry.bind("<Return>", lambda e: self.jump_to_frame())
        ttk.Button(control_frame, text="Go", command=self.jump_to_frame).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=5
        )

        self.mode_var = tk.StringVar(value="ADD")
        ttk.Radiobutton(
            control_frame,
            text="Auto Add",
            variable=self.mode_var,
            value="ADD",
            command=self.change_mode,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(
            control_frame,
            text="Polygon Add",
            variable=self.mode_var,
            value="POLY_ADD",
            command=self.change_mode,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(
            control_frame,
            text="Polygon Erase",
            variable=self.mode_var,
            value="POLY_ERASE",
            command=self.change_mode,
        ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=5
        )

        self.auto_prop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            control_frame, text="Auto-Track on Edit", variable=self.auto_prop_var
        ).pack(side=tk.LEFT, padx=2)

        self.iou_thresh_var = tk.DoubleVar(value=0.40)
        ttk.Label(control_frame, text="Thresh:").pack(side=tk.LEFT, padx=(5, 0))
        ttk.Spinbox(
            control_frame,
            from_=0.05,
            to=1.0,
            increment=0.05,
            width=5,
            textvariable=self.iou_thresh_var,
        ).pack(side=tk.LEFT)

        ttk.Button(control_frame, text="💾 Save JSON", command=self.save_output).pack(
            side=tk.RIGHT, padx=5
        )

        self.lbl_status = ttk.Label(left_frame, text="", font=("Helvetica", 11, "bold"))
        self.lbl_status.pack(side=tk.TOP, anchor=tk.W, padx=5)

        self.canvas = tk.Canvas(left_frame, bg="black", cursor="crosshair")
        self.canvas.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        # Mouse Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind(
            "<ButtonPress-2>", self.on_right_click
        )  # Mac trackpad right-click
        self.canvas.bind(
            "<ButtonPress-3>", self.on_right_click
        )  # Standard external mouse right-click

        self.canvas.bind("<Configure>", self.on_resize)

        # Keyboard Bindings
        self.root.bind("<Key>", self.handle_keypress)

        # Mac Shortcuts for Save and Undo
        self.root.bind("<Command-s>", self.save_output_event)
        self.root.bind("<Command-S>", self.save_output_event)
        self.root.bind("<Command-z>", self.undo_last_action)
        self.root.bind("<Command-Z>", self.undo_last_action)

        # Right Side: Object Manager
        right_frame = ttk.Frame(self.main_paned, padding="10", relief=tk.RIDGE)
        self.main_paned.add(right_frame, weight=1)

        ttk.Label(
            right_frame, text="Objects in Video", font=("Helvetica", 12, "bold")
        ).pack(side=tk.TOP, pady=(0, 10))
        ttk.Button(
            right_frame, text="➕ Add New Object", command=self.add_new_object
        ).pack(side=tk.TOP, fill=tk.X, pady=5)

        ttk.Button(
            right_frame, text="✏️ Edit Object Name", command=self.edit_object_name
        ).pack(side=tk.TOP, fill=tk.X, pady=2)

        ttk.Button(
            right_frame, text="🗑️ Delete Object", command=self.delete_object
        ).pack(side=tk.TOP, fill=tk.X, pady=2)

        self.obj_listbox = tk.Listbox(
            right_frame,
            selectmode=tk.SINGLE,
            font=("Helvetica", 11),
            exportselection=False,
        )
        self.obj_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.obj_listbox.bind("<<ListboxSelect>>", self.on_object_select)

        ttk.Button(
            right_frame,
            text="Clear Active Mask (C)",
            command=self.clear_active_mask,
        ).pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    def load_cropformer_masks(self, json_path):
        num_frames = len(self.frames)
        proposals_per_frame = [[] for _ in range(num_frames)]
        if not os.path.exists(json_path):
            print(f"    --- Warning: Could not find {json_path}")
            return proposals_per_frame

        with open(json_path, "r") as f:
            raw_json = json.load(f)
            data = raw_json.get("all_tracks", raw_json)

        for frame_str, objects in data.items():
            frame_idx = int(frame_str)
            if 0 <= frame_idx < num_frames:
                for oid, rle in objects.items():
                    if rle:
                        proposals_per_frame[frame_idx].append(rle)
        return proposals_per_frame

    def get_active_proposals(self, frame_idx=None):
        if frame_idx is None:
            frame_idx = self.frame_idx

        active_rles = self.proposals[frame_idx]
        decoded_masks = []

        for rle in active_rles:
            bmask = rle_to_bmask(rle)
            decoded_masks.append((bmask * 255).astype(np.uint8))

        return decoded_masks

    def load_existing_annotations(self):
        in_file = osp.join("annotations", f"{self.instance_name}_annotations.json")
        if not os.path.exists(in_file):
            return

        print(f">>> Found existing annotations at {in_file}. Loading into tool...")
        with open(in_file, "r") as f:
            data = json.load(f)

        max_obj_id = -1
        for frame_str, objs in data.items():
            frame_idx = int(frame_str)
            if not (0 <= frame_idx < len(self.frames)):
                continue

            for obj_id_str, obj_data in objs.items():
                obj_id = int(obj_id_str)
                name = obj_data.get("object_name", f"Object_{obj_id}")

                max_obj_id = max(max_obj_id, obj_id)

                if obj_id not in self.objects:
                    self.objects[obj_id] = name
                    self.object_masks[obj_id] = np.zeros(
                        (len(self.frames), self.orig_H, self.orig_W), dtype=np.uint8
                    )
                    self.bad_masks[obj_id] = np.zeros(len(self.frames), dtype=bool)

                if "counts" in obj_data:
                    bmask = rle_to_bmask(obj_data)
                    self.object_masks[obj_id][frame_idx] = (bmask * 255).astype(
                        np.uint8
                    )

                self.bad_masks[obj_id][frame_idx] = obj_data.get("bad_mask", False)

        if self.objects:
            self.next_obj_id = max_obj_id + 1
            for obj_id, name in sorted(self.objects.items()):
                display_text = f"[{obj_id}] {name}"
                self.obj_listbox.insert(tk.END, display_text)

            self.obj_listbox.selection_set(0)
            self.on_object_select(None)

    def add_new_object(self):
        name = simpledialog.askstring(
            "New Object", "Enter object name:", parent=self.root
        )
        if not name:
            return

        obj_id = self.next_obj_id
        self.next_obj_id += 1

        self.objects[obj_id] = name
        self.object_masks[obj_id] = np.zeros(
            (len(self.frames), self.orig_H, self.orig_W), dtype=np.uint8
        )
        self.bad_masks[obj_id] = np.zeros(len(self.frames), dtype=bool)

        display_text = f"[{obj_id}] {name}"
        self.obj_listbox.insert(tk.END, display_text)
        self.obj_listbox.selection_clear(0, tk.END)
        self.obj_listbox.selection_set(tk.END)
        self.on_object_select(None)

    def edit_object_name(self):
        if self.active_obj_id is None:
            messagebox.showwarning("Warning", "Please select an object to edit first.")
            return

        current_name = self.objects[self.active_obj_id]
        new_name = simpledialog.askstring(
            "Edit Object Name",
            "Enter new object name:",
            initialvalue=current_name,
            parent=self.root,
        )

        if new_name and new_name.strip() != "":
            self.objects[self.active_obj_id] = new_name.strip()

            items = self.obj_listbox.get(0, tk.END)
            for i, item in enumerate(items):
                if item.startswith(f"[{self.active_obj_id}]"):
                    self.obj_listbox.delete(i)
                    self.obj_listbox.insert(
                        i, f"[{self.active_obj_id}] {new_name.strip()}"
                    )
                    self.obj_listbox.selection_set(i)
                    break

            self.update_view()

    def delete_object(self):
        if self.active_obj_id is None:
            messagebox.showwarning(
                "Warning", "Please select an object to delete first."
            )
            return

        obj_name = self.objects[self.active_obj_id]
        confirm = messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to completely delete the object '{obj_name}' (ID: {self.active_obj_id}) and all its masks?",
        )

        if confirm:
            self.poly_points = []
            self.undo_state = None  # Clear undo state on delete to prevent errors

            del self.objects[self.active_obj_id]
            del self.object_masks[self.active_obj_id]
            del self.bad_masks[self.active_obj_id]

            items = self.obj_listbox.get(0, tk.END)
            for i, item in enumerate(items):
                if item.startswith(f"[{self.active_obj_id}]"):
                    self.obj_listbox.delete(i)
                    break

            self.active_obj_id = None
            self.update_view()

    # Undo / State management
    def save_state_for_undo(self):
        if self.active_obj_id is not None:
            self.undo_state = {
                "frame_idx": self.frame_idx,
                "obj_id": self.active_obj_id,
                "mask": self.object_masks[self.active_obj_id][self.frame_idx].copy(),
                "bad_mask": bool(self.bad_masks[self.active_obj_id][self.frame_idx]),
                "poly_points": list(self.poly_points),
                "mask_modified": self.mask_modified,
            }

    def undo_last_action(self, event=None):
        if self.undo_state is None:
            return "break"  # Nothing to undo

        frame = self.undo_state["frame_idx"]
        obj = self.undo_state["obj_id"]

        # Re-navigate if the user moved to a different frame/object before pressing Undo
        if self.frame_idx != frame:
            self.frame_idx = frame
        if self.active_obj_id != obj:
            self.active_obj_id = obj

            # Sync UI listbox selection
            items = self.obj_listbox.get(0, tk.END)
            for i, item in enumerate(items):
                if item.startswith(f"[{obj}]"):
                    self.obj_listbox.selection_clear(0, tk.END)
                    self.obj_listbox.selection_set(i)
                    break

        # Restore state
        self.object_masks[obj][frame] = self.undo_state["mask"]
        self.bad_masks[obj][frame] = self.undo_state["bad_mask"]
        self.poly_points = self.undo_state["poly_points"]
        self.mask_modified = self.undo_state["mask_modified"]

        self.undo_state = None  # Enforce 1-step undo only
        self.update_view()
        return "break"

    def propagate_mask(self, from_idx, to_idx, force_overwrite=False):
        if not self.auto_prop_var.get() or self.active_obj_id is None:
            return False

        target_mask = self.object_masks[self.active_obj_id][to_idx]
        if np.any(target_mask) and not force_overwrite:
            return False

        source_mask = self.object_masks[self.active_obj_id][from_idx]
        if not np.any(source_mask):
            if force_overwrite:
                target_mask.fill(0)
            return False

        proposals = self.get_active_proposals(frame_idx=to_idx)
        if not proposals:
            if force_overwrite:
                target_mask.fill(0)
            return False

        try:
            thresh = float(self.iou_thresh_var.get())
        except ValueError:
            thresh = 0.4

        accumulated_mask = np.zeros_like(target_mask)
        added_any = False

        for prop_mask in proposals:
            intersection = cv2.bitwise_and(source_mask, prop_mask)
            i_sum = np.count_nonzero(intersection)
            if i_sum == 0:
                continue

            u_sum = np.count_nonzero(cv2.bitwise_or(source_mask, prop_mask))
            iou = i_sum / u_sum

            p_sum = np.count_nonzero(prop_mask)
            coverage = i_sum / p_sum if p_sum > 0 else 0

            if iou >= thresh or coverage >= 0.75:
                accumulated_mask = cv2.bitwise_or(accumulated_mask, prop_mask)
                added_any = True

        if added_any:
            self.object_masks[self.active_obj_id][to_idx] = accumulated_mask
            self.bad_masks[self.active_obj_id][to_idx] = False
            return True
        else:
            if force_overwrite:
                target_mask.fill(0)
            return False

    def _run_forward_propagation(self):
        self.mask_modified = False
        if not self.auto_prop_var.get() or self.active_obj_id is None:
            return

        for i in range(self.frame_idx, len(self.frames) - 1):
            success = self.propagate_mask(
                from_idx=i, to_idx=i + 1, force_overwrite=True
            )
            if not success:
                break

    def on_object_select(self, event):
        selection = self.obj_listbox.curselection()
        if selection:
            self.mask_modified = False
            text = self.obj_listbox.get(selection[0])
            self.active_obj_id = int(text.split("]")[0][1:])
            self.poly_points = []
            self.update_view()

    def handle_keypress(self, event):
        if self.root.focus_get() == self.frame_entry:
            return

        k = event.keysym.lower()
        if k == "a":
            self.prev_frame()
        elif k == "d":
            self.next_frame()
        elif k == "f":
            self.mark_bad_and_next()
        elif k == "u":
            self.unmark_bad()
        elif k == "c":
            self.clear_active_mask()
        elif k == "return":
            self.finalize_polygon()
        elif k == "escape":
            self.poly_points = []
            self.update_view()

    def mark_bad_and_next(self):
        if self.active_obj_id is not None:
            self.bad_masks[self.active_obj_id][self.frame_idx] = True
        self.next_frame()

    def unmark_bad(self):
        if self.active_obj_id is not None:
            self.bad_masks[self.active_obj_id][self.frame_idx] = False
            self.update_view()

    def jump_to_frame(self):
        try:
            target = int(self.frame_jump_var.get())
            if 0 <= target < len(self.frames):
                self.poly_points = []
                self.mask_modified = False
                self.frame_idx = target
                self.update_view()
                self.root.focus_set()
            else:
                messagebox.showwarning(
                    "Invalid Frame",
                    f"Please enter a frame between 0 and {len(self.frames)-1}.",
                )
                self.frame_jump_var.set(str(self.frame_idx))
        except ValueError:
            messagebox.showwarning(
                "Invalid Input", "Please enter a valid frame number."
            )
            self.frame_jump_var.set(str(self.frame_idx))

    def prev_frame(self):
        self.poly_points = []
        self.mask_modified = False

        if self.frame_idx > 0:
            self.frame_idx -= 1
        self.update_view()

    def next_frame(self):
        self.poly_points = []

        if self.active_obj_id is not None and self.frame_idx < len(self.frames) - 1:
            success = self.propagate_mask(
                self.frame_idx, self.frame_idx + 1, force_overwrite=self.mask_modified
            )

            if self.mask_modified and success:
                self.mask_modified = True
            else:
                self.mask_modified = False

        if self.frame_idx < len(self.frames) - 1:
            self.frame_idx += 1

        self.update_view()

    def change_mode(self):
        self.poly_points = []
        self.mode = self.mode_var.get()
        self.update_view()

    def clear_active_mask(self):
        if self.active_obj_id is not None:
            self.save_state_for_undo()  # Backup state before clearing

            self.poly_points = []
            self.object_masks[self.active_obj_id][self.frame_idx].fill(0)
            self.bad_masks[self.active_obj_id][self.frame_idx] = False
            self.mask_modified = True
            self.update_view()

    def get_img_coords(self, event_x, event_y, clamp=True):
        x = int((event_x - self.display_offset_x) / self.display_scale)
        y = int((event_y - self.display_offset_y) / self.display_scale)

        if clamp:
            return max(0, min(self.orig_W - 1, x)), max(0, min(self.orig_H - 1, y))
        return x, y

    def on_mouse_down(self, event):
        if self.active_obj_id is None:
            messagebox.showwarning("Warning", "Please add and select an object first.")
            return

        self.save_state_for_undo()  # Backup state before adding dots/segments
        x, y = self.get_img_coords(event.x, event.y, clamp=True)

        if self.mode == "ADD":
            for mask in self.get_active_proposals():
                if mask[y, x] > 0:
                    self.object_masks[self.active_obj_id][self.frame_idx] = (
                        cv2.bitwise_or(
                            self.object_masks[self.active_obj_id][self.frame_idx], mask
                        )
                    )
                    self.bad_masks[self.active_obj_id][self.frame_idx] = False
                    self.mask_modified = True
            self.update_view()
        elif self.mode in ["POLY_ADD", "POLY_ERASE"]:
            x, y = self.get_img_coords(event.x, event.y, clamp=False)
            self.poly_points.append((x, y))
            self.update_view()

    def on_right_click(self, event):
        if self.mode in ["POLY_ADD", "POLY_ERASE"]:
            self.finalize_polygon()

    def finalize_polygon(self):
        if self.active_obj_id is None or len(self.poly_points) < 3:
            self.poly_points = []
            self.update_view()
            return

        self.save_state_for_undo()  # Backup state before finalizing fill/erase
        pts = np.array(self.poly_points, np.int32).reshape((-1, 1, 2))

        if self.mode == "POLY_ADD":
            cv2.fillPoly(
                self.object_masks[self.active_obj_id][self.frame_idx], [pts], 255
            )
            self.bad_masks[self.active_obj_id][self.frame_idx] = False
            self.mask_modified = True
        elif self.mode == "POLY_ERASE":
            cv2.fillPoly(
                self.object_masks[self.active_obj_id][self.frame_idx], [pts], 0
            )
            self.bad_masks[self.active_obj_id][self.frame_idx] = False
            self.mask_modified = True

        self.poly_points = []
        self.update_view()

    def on_resize(self, event):
        self.update_view()

    def update_view(self):
        img = cv2.imread(self.frames[self.frame_idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay = img.copy()

        is_bad_mask = False
        if self.active_obj_id is not None:
            mask = self.object_masks[self.active_obj_id][self.frame_idx]
            is_bad_mask = self.bad_masks[self.active_obj_id][self.frame_idx]

            if np.any(mask):
                if is_bad_mask:
                    overlay[mask > 0] = [255, 100, 0]  # Orange for Bad Mask
                else:
                    overlay[mask > 0] = [0, 255, 0]  # Green for Good Mask

        blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

        self.canvas.update_idletasks()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()

        if cw > 10 and ch > 10:
            pad = 100
            avail_cw = max(10, cw - 2 * pad)
            avail_ch = max(10, ch - 2 * pad)

            self.display_scale = min(avail_cw / self.orig_W, avail_ch / self.orig_H)
            new_W = int(self.orig_W * self.display_scale)
            new_H = int(self.orig_H * self.display_scale)

            self.display_offset_x = (cw - new_W) // 2
            self.display_offset_y = (ch - new_H) // 2
            blended = cv2.resize(blended, (new_W, new_H), interpolation=cv2.INTER_AREA)

        img_pil = Image.fromarray(blended)
        self.photo_img = ImageTk.PhotoImage(image=img_pil)
        self.canvas.delete("all")

        self.canvas.create_image(
            self.display_offset_x,
            self.display_offset_y,
            anchor=tk.NW,
            image=self.photo_img,
        )

        self.canvas.create_rectangle(
            self.display_offset_x,
            self.display_offset_y,
            self.display_offset_x + new_W,
            self.display_offset_y + new_H,
            outline="gray",
            width=1,
        )

        if len(self.poly_points) > 0:
            canvas_pts = []
            for px, py in self.poly_points:
                cx = self.display_offset_x + px * self.display_scale
                cy = self.display_offset_y + py * self.display_scale
                canvas_pts.append((cx, cy))

                r = 4
                self.canvas.create_oval(
                    cx - r, cy - r, cx + r, cy + r, fill="red", outline="red"
                )

            if len(canvas_pts) > 1:
                flat_coords = [coord for pt in canvas_pts for coord in pt]
                self.canvas.create_line(flat_coords, fill="red", width=2)

        active_name = self.objects.get(self.active_obj_id, "None")
        bad_text = " [BAD MASK]" if is_bad_mask else ""

        hint = (
            " | Hint: Click points, Right-Click to fill, Cmd+Z to Undo, Cmd+S to Save"
            if self.mode in ["POLY_ADD", "POLY_ERASE"]
            else " | Cmd+Z to Undo, Cmd+S to Save"
        )

        self.lbl_status.config(
            text=f"Frame: {self.frame_idx}/{len(self.frames)-1} | Active Object: [{self.active_obj_id}] {active_name}{bad_text}{hint}"
        )

        self.frame_jump_var.set(str(self.frame_idx))

    def save_output_event(self, event=None):
        self.save_output()
        return "break"

    def save_output(self):
        self.mask_modified = False

        output_dict = {}
        for frame_idx in range(len(self.frames)):
            frame_str = str(frame_idx)
            output_dict[frame_str] = {}
            for obj_id, name in self.objects.items():
                mask = self.object_masks[obj_id][frame_idx]
                if np.any(mask):
                    rle = bmask_to_rle(mask > 0)
                    if isinstance(rle.get("counts"), bytes):
                        rle["counts"] = rle["counts"].decode("utf-8")
                    output_dict[frame_str][str(obj_id)] = {
                        "object_name": name,
                        "counts": rle["counts"],
                        "size": rle["size"],
                        "bad_mask": bool(self.bad_masks[obj_id][frame_idx]),
                    }

        save_dir = "annotations"
        os.makedirs(save_dir, exist_ok=True)
        out_file = osp.join(save_dir, f"{self.instance_name}_annotations.json")

        with open(out_file, "w") as f:
            json.dump(output_dict, f, indent=4)

        messagebox.showinfo("Success", f"All objects saved to {out_file}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Mask Annotation Tool")
    p.add_argument("-i", "--instance", required=True, help="Instance name")

    # Make frames_dir optional by setting a default to None
    p.add_argument(
        "--frames_dir",
        default=None,
        help="Path to video frames (auto-inferred if not provided)",
    )
    p.add_argument(
        "--json_dir",
        default="_interm_out/entities_custom_cropformer",
        help="Directory with CropFormer JSONs",
    )
    args = p.parse_args()

    # Automatically infer frames_dir if it wasn't passed via command line
    if args.frames_dir is None:
        args.frames_dir = os.path.join("assets", "example", args.instance)

    root = tk.Tk()
    root.geometry("1200x800")
    app = AnnotationToolApp(root, args.frames_dir, args.json_dir, args.instance)
    root.mainloop()

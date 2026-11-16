"""
Usage:
python dataset/transformation_tool.py -i banana
python dataset/transformation_tool.py -i banana --frames_dir custom/path/banana --annotations_path custom/banana_annotations.json

"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import os
import glob
import sys
import os.path as osp
import colorsys

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))
from utils import rle_to_bmask

GLOBAL_PALETTE = (
    np.array([colorsys.hsv_to_rgb(i / 30.0, 1, 1) for i in range(30)]) * 255
)


class TransformationToolApp:
    def __init__(self, root, frames_dir, annotations_path, instance_name):
        self.root = root
        self.root.title(f"Transformation Tool - Instance: {instance_name}")

        self.frames_dir = frames_dir
        self.annotations_path = annotations_path
        self.instance_name = instance_name
        self.save_dir = "annotations"
        self.trans_out_file = osp.join(
            self.save_dir, f"{self.instance_name}_transformations.json"
        )

        # Load Frames
        self.frames = sorted(glob.glob(os.path.join(self.frames_dir, "*.jpg")))
        if not self.frames:
            messagebox.showerror("Error", f"No frames found in {self.frames_dir}")
            sys.exit(1)

        sample_frame = cv2.imread(self.frames[0])
        self.orig_H, self.orig_W = sample_frame.shape[:2]

        # Load Ground Truth Annotations
        print(f">>> Loading object annotations from {self.annotations_path}...")
        if not os.path.exists(self.annotations_path):
            messagebox.showerror(
                "Error", f"Annotations file not found: {self.annotations_path}"
            )
            sys.exit(1)

        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)

        # Extract all objects known in the video
        self.known_objects = {}
        for frame_str, objs in self.annotations.items():
            for obj_id_str, data in objs.items():
                self.known_objects[int(obj_id_str)] = data["object_name"]

        # App State
        self.frame_idx = 0
        self.transformations = []
        self.active_trans_idx = None

        # State for active renaming
        self.active_rename_side = None
        self.active_rename_id = None

        # UI State for resizing
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0

        self.setup_ui()

        # Load existing transformations if you are coming back to revise
        self.load_existing_transformations()

        self.update_view()

    def setup_ui(self):
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # --- Left Side: Video Player ---
        left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(left_frame, weight=5)

        control_frame = ttk.Frame(left_frame, padding="5")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(control_frame, text="Prev (A)", command=self.prev_frame).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(control_frame, text="Next (D)", command=self.next_frame).pack(
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

        self.lbl_status = ttk.Label(left_frame, text="", font=("Helvetica", 11, "bold"))
        self.lbl_status.pack(side=tk.TOP, anchor=tk.W, padx=5)

        self.canvas = tk.Canvas(left_frame, bg="black")
        self.canvas.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        self.canvas.bind("<Configure>", self.on_resize)
        self.root.bind("<Key>", self.handle_keypress)

        # --- Right Side: Transformation Manager ---
        right_frame = ttk.Frame(self.main_paned, padding="10", relief=tk.RIDGE)
        self.main_paned.add(right_frame, weight=3)

        ttk.Label(
            right_frame, text="Transformations", font=("Helvetica", 14, "bold")
        ).pack(side=tk.TOP, pady=(0, 10))
        ttk.Button(
            right_frame,
            text="➕ Initiate New Transformation",
            command=self.add_transformation,
        ).pack(side=tk.TOP, fill=tk.X, pady=5)
        ttk.Button(
            right_frame, text="🗑️ Delete Selected", command=self.delete_transformation
        ).pack(side=tk.TOP, fill=tk.X, pady=2)

        self.trans_listbox = tk.Listbox(
            right_frame,
            selectmode=tk.SINGLE,
            height=6,
            font=("Helvetica", 11),
            exportselection=False,
        )
        self.trans_listbox.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.trans_listbox.bind("<<ListboxSelect>>", self.on_trans_select)

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(
            side=tk.TOP, fill=tk.X, pady=10
        )

        # --- Transformation Editor Form ---
        self.form_frame = ttk.Frame(right_frame)
        self.form_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        f_frames = ttk.Frame(self.form_frame)
        f_frames.pack(fill=tk.X, pady=2)
        self.lbl_start_frame = ttk.Label(f_frames, text="Start Frame: None")
        self.lbl_start_frame.pack(side=tk.LEFT, padx=2)
        ttk.Button(
            f_frames, text="Set Current", command=lambda: self.set_frame("start")
        ).pack(side=tk.LEFT)

        self.lbl_end_frame = ttk.Label(f_frames, text="End Frame: None")
        self.lbl_end_frame.pack(side=tk.LEFT, padx=(15, 2))
        ttk.Button(
            f_frames, text="Set Current", command=lambda: self.set_frame("end")
        ).pack(side=tk.LEFT)

        f_type = ttk.Frame(self.form_frame)
        f_type.pack(fill=tk.X, pady=5)
        ttk.Label(f_type, text="Type:").pack(side=tk.LEFT)
        self.type_var = tk.StringVar(value="decomposition")
        ttk.Radiobutton(
            f_type, text="Decomposition", variable=self.type_var, value="decomposition"
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            f_type, text="Composition", variable=self.type_var, value="composition"
        ).pack(side=tk.LEFT, padx=5)

        f_objs = ttk.Frame(self.form_frame)
        f_objs.pack(fill=tk.BOTH, expand=True, pady=5)

        f_start_objs = ttk.Frame(f_objs)
        f_start_objs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        ttk.Label(f_start_objs, text="Start State:").pack(anchor=tk.W)
        self.lb_start_objs = tk.Listbox(
            f_start_objs, selectmode=tk.MULTIPLE, height=5, exportselection=False
        )
        self.lb_start_objs.pack(fill=tk.BOTH, expand=True)

        f_end_objs = ttk.Frame(f_objs)
        f_end_objs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        ttk.Label(f_end_objs, text="End State:").pack(anchor=tk.W)
        self.lb_end_objs = tk.Listbox(
            f_end_objs, selectmode=tk.MULTIPLE, height=5, exportselection=False
        )
        self.lb_end_objs.pack(fill=tk.BOTH, expand=True)

        self.lb_start_objs.bind("<<ListboxSelect>>", self.on_obj_list_select)
        self.lb_end_objs.bind("<<ListboxSelect>>", self.on_obj_list_select)

        self.populate_object_listboxes()

        # --- Inline Rename Form ---
        f_edit_name = ttk.Frame(self.form_frame)
        f_edit_name.pack(fill=tk.X, pady=5)

        ttk.Label(f_edit_name, text="Target:").grid(
            row=0, column=0, padx=2, sticky=tk.W
        )
        self.lbl_selected_obj_id = ttk.Label(f_edit_name, text="None", width=12)
        self.lbl_selected_obj_id.grid(row=0, column=1, padx=2, sticky=tk.W)

        ttk.Label(f_edit_name, text="Name:").grid(row=0, column=2, padx=2, sticky=tk.W)
        self.entry_obj_name = ttk.Entry(f_edit_name, state="disabled")
        self.entry_obj_name.grid(row=0, column=3, padx=2, sticky="ew")

        self.btn_apply_name = ttk.Button(
            f_edit_name,
            text="Apply Override",
            state="disabled",
            command=self.apply_name_change,
        )
        self.btn_apply_name.grid(row=0, column=4, padx=2)

        f_edit_name.columnconfigure(3, weight=1)

        # --- Description & Export ---
        f_desc = ttk.Frame(self.form_frame)
        f_desc.pack(fill=tk.X, pady=5)
        ttk.Label(f_desc, text="Description:").pack(anchor=tk.W)
        self.entry_desc = ttk.Entry(f_desc)
        self.entry_desc.pack(fill=tk.X)

        ttk.Button(
            self.form_frame,
            text="✅ Update Active Transformation",
            command=self.save_form_to_active,
        ).pack(fill=tk.X, pady=10)
        ttk.Button(
            right_frame, text="💾 Export Transformations JSON", command=self.export_json
        ).pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        self.disable_form()

    # --- Interaction & Form Management ---
    def load_existing_transformations(self):
        if os.path.exists(self.trans_out_file):
            print(
                f">>> Found existing transformations at {self.trans_out_file}. Loading into tool..."
            )
            with open(self.trans_out_file, "r") as f:
                self.transformations = json.load(f)
            self.update_listbox_titles()

    def add_transformation(self):
        new_id = (
            max([t.get("id", -1) for t in self.transformations]) + 1
            if self.transformations
            else 0
        )

        new_trans = {
            "id": new_id,
            "start_frame": self.frame_idx,
            "end_frame": self.frame_idx,
            "type": "decomposition",
            "start_state_objects": [],
            "end_state_objects": [],
            "description": "",
            "start_names": {},
            "end_names": {},
        }
        self.transformations.append(new_trans)
        self.update_listbox_titles()

        self.trans_listbox.selection_clear(0, tk.END)
        self.trans_listbox.selection_set(tk.END)
        self.on_trans_select(None)

    def delete_transformation(self):
        if self.active_trans_idx is None:
            return

        confirm = messagebox.askyesno(
            "Confirm Delete", "Are you sure you want to delete this transformation?"
        )
        if confirm:
            del self.transformations[self.active_trans_idx]
            self.active_trans_idx = None
            self.update_listbox_titles()
            self.disable_form()
            self.update_view()

    def on_trans_select(self, event):
        selection = self.trans_listbox.curselection()
        if not selection:
            self.disable_form()
            self.update_view()
            return

        self.active_trans_idx = selection[0]
        self.enable_form()
        self.load_active_to_form()

        # Reset the rename form
        self.active_rename_side = None
        self.active_rename_id = None
        self.lbl_selected_obj_id.config(text="None")
        self.entry_obj_name.delete(0, tk.END)
        self.entry_obj_name.config(state="disabled")
        self.btn_apply_name.config(state="disabled")

    def set_frame(self, target):
        if self.active_trans_idx is None:
            return

        if target == "start":
            self.lbl_start_frame.config(text=f"Start Frame: {self.frame_idx}")
            self.transformations[self.active_trans_idx]["start_frame"] = self.frame_idx
        elif target == "end":
            self.lbl_end_frame.config(text=f"End Frame: {self.frame_idx}")
            self.transformations[self.active_trans_idx]["end_frame"] = self.frame_idx

        self.update_listbox_titles()

    def populate_object_listboxes(self):
        self.lb_start_objs.delete(0, tk.END)
        self.lb_end_objs.delete(0, tk.END)

        trans = (
            self.transformations[self.active_trans_idx]
            if self.active_trans_idx is not None
            else None
        )

        start_overrides = trans.get("start_names", {}) if trans else {}
        end_overrides = trans.get("end_names", {}) if trans else {}

        for obj_id, default_name in sorted(self.known_objects.items()):
            # Start Side string
            s_name = start_overrides.get(str(obj_id), default_name)
            self.lb_start_objs.insert(tk.END, f"[{obj_id}] {s_name}")

            # End Side string
            e_name = end_overrides.get(str(obj_id), default_name)
            self.lb_end_objs.insert(tk.END, f"[{obj_id}] {e_name}")

    def load_active_to_form(self):
        trans = self.transformations[self.active_trans_idx]

        self.lbl_start_frame.config(text=f"Start Frame: {trans['start_frame']}")
        self.lbl_end_frame.config(text=f"End Frame: {trans['end_frame']}")
        self.type_var.set(trans["type"])

        self.entry_desc.delete(0, tk.END)
        self.entry_desc.insert(0, trans["description"])

        # Backwards compatibility check & extract IDs
        start_data = trans.get("start_state_objects", [])
        start_ids = [obj["id"] if isinstance(obj, dict) else obj for obj in start_data]

        end_data = trans.get("end_state_objects", [])
        end_ids = [obj["id"] if isinstance(obj, dict) else obj for obj in end_data]

        # Sync local independent overrides in case JSON was loaded from disk
        if "start_names" not in trans:
            trans["start_names"] = {}
        if "end_names" not in trans:
            trans["end_names"] = {}

        for obj in start_data:
            if isinstance(obj, dict):
                trans["start_names"][str(obj["id"])] = obj["name"]

        for obj in end_data:
            if isinstance(obj, dict):
                trans["end_names"][str(obj["id"])] = obj["name"]

        self.populate_object_listboxes()

        self.lb_start_objs.selection_clear(0, tk.END)
        for i, obj_id in enumerate(sorted(self.known_objects.keys())):
            if obj_id in start_ids:
                self.lb_start_objs.selection_set(i)

        self.lb_end_objs.selection_clear(0, tk.END)
        for i, obj_id in enumerate(sorted(self.known_objects.keys())):
            if obj_id in end_ids:
                self.lb_end_objs.selection_set(i)

        self.update_view()

    def on_obj_list_select(self, event=None):
        self.update_view()

        if self.active_trans_idx is None:
            return

        start_sel = self.lb_start_objs.curselection()
        end_sel = self.lb_end_objs.curselection()

        # Only allow rename if exactly one side has a single selection and the other has none
        if len(start_sel) == 1 and len(end_sel) == 0:
            self.active_rename_side = "start"
            idx = start_sel[0]
            obj_id = sorted(self.known_objects.keys())[idx]
        elif len(start_sel) == 0 and len(end_sel) == 1:
            self.active_rename_side = "end"
            idx = end_sel[0]
            obj_id = sorted(self.known_objects.keys())[idx]
        else:
            self.active_rename_side = None
            obj_id = None

        if obj_id is not None:
            self.active_rename_id = obj_id
            trans = self.transformations[self.active_trans_idx]

            overrides = trans.get(f"{self.active_rename_side}_names", {})
            current_name = overrides.get(str(obj_id), self.known_objects[obj_id])

            self.lbl_selected_obj_id.config(
                text=f"{self.active_rename_side.capitalize()} ID: {obj_id}"
            )
            self.entry_obj_name.config(state="normal")
            self.entry_obj_name.delete(0, tk.END)
            self.entry_obj_name.insert(0, current_name)
            self.btn_apply_name.config(state="normal")
        else:
            self.active_rename_id = None
            self.lbl_selected_obj_id.config(text="None")
            self.entry_obj_name.delete(0, tk.END)
            self.entry_obj_name.config(state="disabled")
            self.btn_apply_name.config(state="disabled")

    def apply_name_change(self):
        if self.active_rename_id is None or self.active_rename_side is None:
            return

        new_name = self.entry_obj_name.get().strip()
        trans = self.transformations[self.active_trans_idx]

        dict_key = f"{self.active_rename_side}_names"
        if dict_key not in trans:
            trans[dict_key] = {}

        obj_id_str = str(self.active_rename_id)

        if new_name == "":
            if obj_id_str in trans[dict_key]:
                del trans[dict_key][obj_id_str]
        else:
            trans[dict_key][obj_id_str] = new_name

        # Save selections to restore after re-populating listbox
        start_sel = self.lb_start_objs.curselection()
        end_sel = self.lb_end_objs.curselection()

        self.populate_object_listboxes()

        for i in start_sel:
            self.lb_start_objs.selection_set(i)
        for i in end_sel:
            self.lb_end_objs.selection_set(i)

        self.save_form_to_active(show_info=False)
        self.entry_obj_name.focus_set()

    def save_form_to_active(self, show_info=True):
        if self.active_trans_idx is None:
            return

        trans = self.transformations[self.active_trans_idx]
        trans["type"] = self.type_var.get()
        trans["description"] = self.entry_desc.get()

        sorted_ids = sorted(self.known_objects.keys())

        start_objs = []
        for i in self.lb_start_objs.curselection():
            obj_id = sorted_ids[i]
            name = trans.get("start_names", {}).get(
                str(obj_id), self.known_objects[obj_id]
            )
            start_objs.append({"id": obj_id, "name": name})

        end_objs = []
        for i in self.lb_end_objs.curselection():
            obj_id = sorted_ids[i]
            name = trans.get("end_names", {}).get(
                str(obj_id), self.known_objects[obj_id]
            )
            end_objs.append({"id": obj_id, "name": name})

        trans["start_state_objects"] = start_objs
        trans["end_state_objects"] = end_objs

        self.update_listbox_titles()
        if show_info:
            messagebox.showinfo("Saved", "Transformation details updated.")

    def update_listbox_titles(self):
        self.trans_listbox.delete(0, tk.END)
        for trans in self.transformations:
            t_type = "Decomp" if trans.get("type") == "decomposition" else "Comp"
            title = f"[{t_type}] ID {trans['id']}: F{trans['start_frame']} -> F{trans['end_frame']}"
            self.trans_listbox.insert(tk.END, title)

        if self.active_trans_idx is not None and self.active_trans_idx < len(
            self.transformations
        ):
            self.trans_listbox.selection_set(self.active_trans_idx)

    def disable_form(self):
        for child in self.form_frame.winfo_children():
            if isinstance(child, ttk.Frame):
                for sub in child.winfo_children():
                    try:
                        sub.configure(state="disabled")
                    except tk.TclError:
                        pass
            else:
                try:
                    child.configure(state="disabled")
                except tk.TclError:
                    pass

    def enable_form(self):
        for child in self.form_frame.winfo_children():
            if isinstance(child, ttk.Frame):
                for sub in child.winfo_children():
                    try:
                        sub.configure(state="normal")
                    except tk.TclError:
                        pass
            else:
                try:
                    child.configure(state="normal")
                except tk.TclError:
                    pass

    # --- Video Player ---
    def handle_keypress(self, event):
        if self.root.focus_get() == self.frame_entry:
            return
        if self.root.focus_get() == self.entry_obj_name:
            if event.keysym == "Return":
                self.apply_name_change()
            return

        k = event.char.lower()
        if k == "a":
            self.prev_frame()
        elif k == "d":
            self.next_frame()

    def jump_to_frame(self):
        try:
            target = int(self.frame_jump_var.get())
            if 0 <= target < len(self.frames):
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
        self.frame_idx = max(0, self.frame_idx - 1)
        self.update_view()

    def next_frame(self):
        self.frame_idx = min(len(self.frames) - 1, self.frame_idx + 1)
        self.update_view()

    def on_resize(self, event):
        self.update_view()

    def update_view(self):
        img = cv2.imread(self.frames[self.frame_idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay = img.copy()

        active_obj_ids = set()
        if self.active_trans_idx is not None:
            sorted_ids = sorted(self.known_objects.keys())
            for idx in self.lb_start_objs.curselection():
                active_obj_ids.add(sorted_ids[idx])
            for idx in self.lb_end_objs.curselection():
                active_obj_ids.add(sorted_ids[idx])

        frame_str = str(self.frame_idx)
        if frame_str in self.annotations:
            for obj_id_str, obj_data in self.annotations[frame_str].items():
                obj_id = int(obj_id_str)

                if obj_id in active_obj_ids:
                    color = GLOBAL_PALETTE[obj_id % len(GLOBAL_PALETTE)]
                    if "counts" in obj_data:
                        bmask = rle_to_bmask(obj_data)
                        overlay[bmask > 0] = color

        blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        self.canvas.update_idletasks()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw > 10 and ch > 10:
            self.display_scale = min(cw / self.orig_W, ch / self.orig_H)
            new_W, new_H = int(self.orig_W * self.display_scale), int(
                self.orig_H * self.display_scale
            )
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

        self.lbl_status.config(
            text=f"Current Frame: {self.frame_idx} / {len(self.frames)-1}"
        )

        self.frame_jump_var.set(str(self.frame_idx))

    def export_json(self):
        if self.active_trans_idx is not None:
            self.save_form_to_active()

        # Create a deep copy to remove internal UI-only keys before saving
        export_data = json.loads(json.dumps(self.transformations))
        for trans in export_data:
            if "start_names" in trans:
                del trans["start_names"]
            if "end_names" in trans:
                del trans["end_names"]
            # Failsafe clean for any old saves
            if "object_names" in trans:
                del trans["object_names"]

        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.trans_out_file, "w") as f:
            json.dump(export_data, f, indent=4)

        messagebox.showinfo(
            "Success", f"Transformations saved to {self.trans_out_file}"
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Transformation Annotation Tool")
    p.add_argument(
        "-i", "--instance", required=True, help="Instance name (e.g. tissue)"
    )
    p.add_argument(
        "--frames_dir",
        default=None,
        help="Path to video frames (auto-inferred if not provided)",
    )
    p.add_argument(
        "--annotations_path",
        default=None,
        help="Path to the ground truth annotations JSON (auto-inferred if not provided)",
    )
    args = p.parse_args()

    if args.frames_dir is None:
        args.frames_dir = os.path.join("assets", "example", args.instance)

    if args.annotations_path is None:
        args.annotations_path = os.path.join(
            "annotations", f"{args.instance}_annotations.json"
        )

    root = tk.Tk()
    root.geometry("1400x800")
    app = TransformationToolApp(
        root, args.frames_dir, args.annotations_path, args.instance
    )
    root.mainloop()

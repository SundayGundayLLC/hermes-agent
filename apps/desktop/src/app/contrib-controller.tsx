import { useStore } from '@nanostores/react'
import { computed } from 'nanostores'
import type { CSSProperties } from 'react'

import { PREVIEW_RAIL_MAX_WIDTH, PREVIEW_RAIL_MIN_WIDTH } from '@/app/chat/right-rail'
import { PALETTE_AREA, type PaletteContribution } from '@/app/command-palette/contrib'
import { type StatusbarItem } from '@/app/shell/statusbar-controls'
import { toggleLayoutEditMode } from '@/components/pane-shell/edit-mode'
import { allPaneIds, group, split } from '@/components/pane-shell/tree/model'
import { LayoutTreeRoot } from '@/components/pane-shell/tree/renderer'
import {
  $layoutTree,
  bindTreeSideVisibility,
  declareDefaultTree,
  mirrorLayoutTree,
  registerPaneCloser,
  resetLayoutTree,
  revealTreePane,
  setTreePaneHidden,
  watchContributedPanes
} from '@/components/pane-shell/tree/store'
import { SidebarProvider } from '@/components/ui/sidebar'
import { discoverBundledPlugins } from '@/contrib/plugins'
import { Slot } from '@/contrib/react/slot'
import { registry } from '@/contrib/registry'
import { discoverRuntimePlugins } from '@/contrib/runtime-loader'
import { LayoutDashboard } from '@/lib/icons'
import { type KeybindContribution, KEYBINDS_AREA } from '@/lib/keybinds/actions'
import { Codecs, persistentAtom } from '@/lib/persisted'
import { toggleKeybindPanel } from '@/store/keybinds'
import {
  $fileBrowserOpen,
  $panesFlipped,
  $sidebarOpen,
  FILE_BROWSER_DEFAULT_WIDTH,
  FILE_BROWSER_MAX_WIDTH,
  FILE_BROWSER_MIN_WIDTH,
  setFileBrowserOpen,
  setSidebarOpen,
  SIDEBAR_DEFAULT_WIDTH,
  SIDEBAR_MAX_WIDTH
} from '@/store/layout'
import { $filePreviewTarget, $previewTarget, closeRightRail } from '@/store/preview'
import { $reviewOpen, closeReview, REVIEW_PANE_ID } from '@/store/review'
import { $currentCwd } from '@/store/session'

import { watchSessionTiles } from './chat/session-tile'
import { FilesPane, LogsPane, PreviewRailPane, ReviewPaneContent } from './contrib-panes'
import { ContribWiring, WiredPane } from './contrib-wiring'
import { $terminalTakeover, setTerminalTakeover } from './right-sidebar/store'

/**
 * Stripped-down app root (bb/contrib-areas) on the layout TREE model, mounting
 * the REAL app surfaces. The title bar and status bar sit OUTSIDE the grid
 * (fixed chrome) but are fully composable: title bar renders `titleBar.left/
 * right` slots; the status bar consumes `statusBar.left/right` DATA
 * contributions (payload = StatusbarItem). Core registers its items through
 * the same calls a plugin would use.
 */

// ---------------------------------------------------------------------------
// Pane contributions. `data.placement` = semantic role for grid presets;
// `data.minWidth/maxWidth/minHeight/maxHeight` = the SAME clamps the app's
// `Pane` props declare — the layout tree sizes zones by weight (percentage)
// but a zone never shrinks/grows past its active pane's clamp.
// Headers are contextual (tree-side): a pane alone in a zone shows no
// header/tab by default; stacked panes show chips. Double-click a zone
// toggles its header either way.
// ---------------------------------------------------------------------------

registry.registerMany([
  {
    id: 'sessions',
    area: 'panes',
    title: 'sessions',
    // Collapsible: leaves the grid on narrow viewports (edge overlay instead).
    data: {
      placement: 'left',
      collapsible: true,
      revealAliases: ['chat-sidebar'],
      width: `${SIDEBAR_DEFAULT_WIDTH}px`,
      minWidth: `${SIDEBAR_DEFAULT_WIDTH}px`,
      maxWidth: `${SIDEBAR_MAX_WIDTH}px`
    },
    render: () => <WiredPane part="sidebar" />
  },
  {
    id: 'workspace',
    area: 'panes',
    title: 'main',
    data: { placement: 'main', minWidth: '22vw', uncloseable: true },
    render: () => <WiredPane part="chatRoutes" />
  },
  {
    id: 'terminal',
    area: 'panes',
    title: 'terminal',
    data: { placement: 'bottom', height: '38vh', minHeight: '7.5rem', maxHeight: '80vh' },
    render: () => <WiredPane part="terminal" />
  },
  {
    id: 'files',
    area: 'panes',
    title: 'files',
    data: {
      placement: 'right',
      collapsible: true,
      revealAliases: ['file-browser'],
      width: FILE_BROWSER_DEFAULT_WIDTH,
      minWidth: FILE_BROWSER_MIN_WIDTH,
      maxWidth: FILE_BROWSER_MAX_WIDTH
    },
    render: () => <FilesPane />
  },
  {
    id: 'preview',
    area: 'panes',
    title: 'preview',
    // The rail brings its OWN tab strip (per-target tabs with close buttons).
    // Exists only while something is previewed — visibility is bound to the
    // preview targets below, like every other self-managed surface.
    data: {
      placement: 'right',
      width: 'clamp(18rem, 36vw, 32rem)',
      minWidth: PREVIEW_RAIL_MIN_WIDTH,
      maxWidth: PREVIEW_RAIL_MAX_WIDTH
    },
    render: () => <PreviewRailPane />
  },
  {
    id: 'review',
    area: 'panes',
    title: 'review',
    // The second right sidebar: hidden until ⌘G ($reviewOpen) — bound below
    // like the other chrome toggles; its zone collapses while hidden.
    data: {
      placement: 'right',
      collapsible: true,
      revealAliases: [REVIEW_PANE_ID],
      width: FILE_BROWSER_DEFAULT_WIDTH,
      minWidth: FILE_BROWSER_MIN_WIDTH,
      maxWidth: FILE_BROWSER_MAX_WIDTH
    },
    render: () => <ReviewPaneContent />
  },
  {
    // Optional chrome — in NO default layout. Adoption stacks it with the
    // terminal; $logsOpen (default off, ⌘K "Toggle logs") reveals it.
    id: 'logs',
    area: 'panes',
    title: 'logs',
    data: { placement: 'bottom', height: '38vh', minHeight: '7.5rem', maxHeight: '80vh' },
    render: () => <LogsPane />
  }
])

// ---------------------------------------------------------------------------
// Chrome contributions. The title bar and status bar are fixed chrome outside
// the grid, composable through these areas. Everything real lives in the real
// components (TitlebarControls / useStatusbarItems). Sample PLUGIN
// contributions don't live here — they're their own files under `src/plugins/`,
// auto-discovered by discoverBundledPlugins() below.
// ---------------------------------------------------------------------------

registry.registerMany([
  // The session-title dropdown (rename/pin/branch/delete) — the real app's
  // chat header, living in the titlebar's center slot over the workspace.
  { id: 'session-title', area: 'titleBar.center', order: 0, render: () => <WiredPane part="sessionTitle" /> },
  // Layout edit mode registers through the SAME declarative surfaces plugins
  // use: a rebindable keybind (collision-checked in the panel) + a ⌘K row
  // whose hotkey hint tracks the live binding.
  {
    id: 'layout.editMode',
    area: KEYBINDS_AREA,
    data: {
      id: 'layout.editMode',
      label: 'Toggle layout edit mode',
      defaults: ['mod+shift+\\'],
      run: toggleLayoutEditMode
    } satisfies KeybindContribution
  },
  {
    id: 'layout.editMode',
    area: PALETTE_AREA,
    data: {
      id: 'layout.editMode',
      label: 'Toggle layout edit mode',
      action: 'layout.editMode',
      icon: LayoutDashboard,
      keywords: ['layout', 'zones', 'panes', 'edit', 'rearrange'],
      run: toggleLayoutEditMode
    } satisfies PaletteContribution
  },
  // The agent's write -> see loop: rescan <hermes home>/desktop-plugins
  // without relaunching (same-id reloads dispose the previous incarnation).
  {
    id: 'plugins.reload',
    area: PALETTE_AREA,
    data: {
      id: 'plugins.reload',
      label: 'Reload desktop plugins',
      keywords: ['plugins', 'reload', 'refresh', 'desktop'],
      run: () => void discoverRuntimePlugins()
    } satisfies PaletteContribution
  },
  {
    id: 'layout.reset',
    area: PALETTE_AREA,
    data: {
      id: 'layout.reset',
      label: 'Reset layout',
      icon: LayoutDashboard,
      keywords: ['layout', 'reset', 'default', 'panes'],
      run: resetLayoutTree
    } satisfies PaletteContribution
  },
  // The keybind panel's non-titlebar door (the keyboard icon is gone).
  {
    id: 'keybinds.panel',
    area: PALETTE_AREA,
    data: {
      id: 'keybinds.panel',
      label: 'Keyboard shortcuts',
      keywords: ['keybinds', 'shortcuts', 'hotkeys', 'keyboard'],
      run: toggleKeybindPanel
    } satisfies PaletteContribution
  }
])

// ---------------------------------------------------------------------------
// Layout presets — CHAT (main) always dominates.
// ---------------------------------------------------------------------------

// The REAL default: sessions left, chat main, and TWO right sidebars in the
// app's column order (main | … | review | file-browser — files outermost).
// Review only exists while ⌘G ($reviewOpen) has it visible; its zone
// collapses to nothing otherwise.
const DEFAULT_TREE = split(
  'row',
  [
    group(['sessions'], { id: 'grp-sessions' }),
    group(['workspace'], { id: 'grp-main' }),
    split(
      'column',
      [
        split(
          'row',
          [group(['review'], { id: 'grp-review' }), group(['files', 'preview'], { id: 'grp-rail' })],
          [1, 1.2],
          'spl-rail'
        ),
        group(['terminal'], { id: 'grp-terminal' })
      ],
      [1.6, 1],
      'spl-right'
    )
  ],
  [1, 3.4, 1.25],
  'spl-root'
)

const FOCUS_TREE = split(
  'row',
  [group(['sessions']), group(['workspace', 'files', 'preview', 'review', 'terminal'])],
  [1, 4.6]
)

const TERMINAL_TREE = split(
  'column',
  [
    split('row', [group(['sessions']), group(['workspace']), group(['files', 'preview', 'review'])], [1, 3.2, 1.2]),
    group(['terminal'])
  ],
  [3, 1]
)

const QUAD_TREE = split(
  'column',
  [
    split('row', [group(['sessions', 'files']), group(['workspace'])], [1, 3]),
    split('row', [group(['terminal']), group(['preview', 'review'])], [1.4, 1])
  ],
  [3, 1]
)

registry.registerMany([
  { id: 'default', area: 'layouts', title: 'Default', order: 0, data: DEFAULT_TREE },
  { id: 'focus', area: 'layouts', title: 'Focus', order: 10, data: FOCUS_TREE },
  { id: 'terminal-deck', area: 'layouts', title: 'Terminal deck', order: 20, data: TERMINAL_TREE },
  { id: 'quad', area: 'layouts', title: 'Quad', order: 30, data: QUAD_TREE }
])

declareDefaultTree(DEFAULT_TREE)

// Bundled plugins load AFTER core, so a same-id contribution from a plugin
// deliberately overrides the core default (last writer wins). Third-party
// runtime plugins will flow through the same discovery seam.
discoverBundledPlugins()

// Plugin panes join the tree by their `placement` hint the moment they
// register — incl. runtime plugins arriving seconds after boot.
watchContributedPanes()

// Session tiles: persisted split sessions register panes docked beside main.
watchSessionTiles()

// ---------------------------------------------------------------------------
// Titlebar chrome toggles -> tree. The TitlebarControls buttons keep their
// store semantics ($sidebarOpen / $fileBrowserOpen / $panesFlipped); the tree
// reacts — a hidden pane's zone collapses (content stays mounted), the flip
// toggle mirrors the root row.
// ---------------------------------------------------------------------------

function bindPaneVisibility(
  paneId: string,
  $open: { get(): boolean; listen(fn: (open: boolean) => void): void },
  close?: () => void
) {
  setTreePaneHidden(paneId, !$open.get())
  $open.listen(open => setTreePaneHidden(paneId, !open))

  // The tab menu's Close routes through the owning store (never dismissal),
  // so the pane's toggle buttons stay truthful.
  if (close) {
    registerPaneCloser(paneId, close)
  }
}

// SIDES have one source of truth: the TREE. The legacy $panesFlipped flag is
// DERIVED from where the sessions zone actually sits (TitlebarControls maps
// its left/right buttons through it), so dragging sessions across — or
// applying a mirrored preset — remaps the buttons automatically. The flip
// action (⌘\ / titlebar) mirrors the tree only when they disagree.
const sessionsOnRight = () => {
  const tree = $layoutTree.get()

  if (!tree) {
    return null
  }

  const order = allPaneIds(tree)
  const sessions = order.indexOf('sessions')
  const main = order.indexOf('workspace')

  return sessions >= 0 && main >= 0 ? sessions > main : null
}

$layoutTree.subscribe(() => {
  const flipped = sessionsOnRight()

  if (flipped !== null && flipped !== $panesFlipped.get()) {
    $panesFlipped.set(flipped)
  }
})

$panesFlipped.listen(flipped => {
  const current = sessionsOnRight()

  if (current !== null && current !== flipped) {
    mirrorLayoutTree()
  }
})

// POSITIONAL side toggles (titlebar buttons, ⌘B / ⌘J): $sidebarOpen ≙ the
// LEFT side of the main zone, $fileBrowserOpen ≙ the RIGHT — everything on
// that side hides together, whatever panes have been rearranged there.
bindTreeSideVisibility('left', $sidebarOpen, setSidebarOpen)
bindTreeSideVisibility('right', $fileBrowserOpen, setFileBrowserOpen)

// Workspace-scoped surfaces: the file tree and git diff only mean something
// inside a project. A detached chat (no cwd) hides them — their zones
// collapse and the chat absorbs the width; picking a project brings them
// back. The terminal is NOT workspace-gated: unlike the old shell (where it
// rode the rail's row and vanished with it), its zone stands on its own.
const $hasWorkspace = computed($currentCwd, cwd => Boolean(cwd.trim()))

bindPaneVisibility('files', $hasWorkspace)
// ⌘G — the review sidebar appears/disappears (and comes to the front).
bindPaneVisibility(
  'review',
  computed([$reviewOpen, $hasWorkspace], (open, workspace) => open && workspace),
  closeReview
)
// ⌃` / statusbar toggle — the terminal zone follows takeover instead of
// being forced on (PTYs stay alive while hidden; see PersistentTerminal).
bindPaneVisibility('terminal', $terminalTakeover, () => setTerminalTakeover(false))

// Preview EXISTS only while something is previewed (old-shell semantics:
// closing the last preview tab closes the pane; a new target opens + fronts
// it). Same visibility binding as every other self-managed surface, driven
// by the live targets instead of a toggle.
const $previewVisible = computed([$previewTarget, $filePreviewTarget], (target, fileTarget) =>
  Boolean(target || fileTarget)
)

bindPaneVisibility('preview', $previewVisible, closeRightRail)

// Logs are optional chrome: off by default, toggled from ⌘K, persisted.
const $logsOpen = persistentAtom('hermes.desktop.logsOpen', false, Codecs.bool)

bindPaneVisibility('logs', $logsOpen, () => $logsOpen.set(false))
registry.register({
  id: 'logs.toggle',
  area: PALETTE_AREA,
  data: {
    id: 'logs.toggle',
    label: 'Toggle logs',
    keywords: ['logs', 'agent log', 'tail', 'debug'],
    run: () => $logsOpen.set(!$logsOpen.get())
  } satisfies PaletteContribution
})

// Sessions' visibility is the LEFT-side toggle's job — Close just collapses
// the side (⌘B truthful, titlebar button flips back).
registerPaneCloser('sessions', () => setSidebarOpen(false))
// A NEW target while the pane is already visible still fronts it.
$previewTarget.listen(target => target && revealTreePane('preview'))
$filePreviewTarget.listen(target => target && revealTreePane('preview'))

// ---------------------------------------------------------------------------

export function ContribController() {
  const sidebarOpen = useStore($sidebarOpen)

  return (
    <SidebarProvider
      className="h-screen min-h-0 flex-col bg-background"
      onOpenChange={setSidebarOpen}
      open={sidebarOpen}
      style={{ '--sidebar-width': '100%' } as CSSProperties}
    >
      <ContribWiring>
        <div
          className="flex h-screen min-h-0 w-screen flex-col bg-(--ui-bg-chrome) text-(--ui-text-primary)"
          style={{ '--titlebar-height': '0px' } as CSSProperties}
        >
          {/* Title bar: fixed chrome outside the grid, composable via slots.
              Layout contract (no contribution can break it):
                - a full-bar DRAG BASE underneath (pointer-events-none, like
                  AppShell's drag strips) — everywhere without content drags
                  the window;
                - each slot region is width-fit, no-drag, pointer-events-auto,
                  so every contribution is clickable by construction;
                - left/right padding clears the REAL TitlebarControls clusters
                  (fixed, z-70); center is truly window-centered. */}
          <div
            className="relative flex h-[34px] shrink-0 items-center border-b border-(--ui-stroke-tertiary) text-xs"
            style={{
              paddingLeft: 'calc(var(--titlebar-controls-left, 14px) + 2 * var(--titlebar-control-size, 1.25rem) + 1rem)',
              paddingRight: 'calc(var(--titlebar-tools-right, 0.75rem) + 4 * (var(--titlebar-control-size, 1.25rem) + 0.25rem) + 0.5rem)'
            }}
          >
            {/* Drag strips, AppShell-style: cut to AVOID the fixed control
                clusters instead of overlapping them — Electron's no-drag
                carve-out of fixed/transformed elements is unreliable, so a
                full-bar drag base kills their clicks. In-flow slot content
                still carves via its own no-drag wrapper (the same pattern as
                the app's session-title button). */}
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-y-0 left-0 w-(--titlebar-controls-left,14px) [-webkit-app-region:drag]"
            />
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-y-0 left-[calc(var(--titlebar-controls-left,14px)+(var(--titlebar-control-size,1.25rem)*2)+0.75rem)] right-[calc(var(--titlebar-tools-right,0.75rem)+var(--titlebar-tools-width,5.5rem)+0.75rem)] [-webkit-app-region:drag]"
            />
            <div className="pointer-events-auto relative z-10 flex w-max items-center gap-2 [-webkit-app-region:no-drag]">
              <Slot area="titleBar.left" />
            </div>
            <div className="pointer-events-auto absolute left-1/2 top-1/2 z-10 flex w-max -translate-x-1/2 -translate-y-1/2 items-center gap-2 [-webkit-app-region:no-drag]">
              <Slot area="titleBar.center" />
            </div>
            <div className="pointer-events-auto relative z-10 ml-auto flex w-max items-center gap-2 [-webkit-app-region:no-drag]">
              <Slot area="titleBar.right" />
            </div>
          </div>

          <LayoutTreeRoot />

          {/* The REAL statusbar (model pill, command center, agents, …) with
              statusBar.left/right contributions merged in. */}
          <WiredPane part="statusbar" />
        </div>
      </ContribWiring>
    </SidebarProvider>
  )
}

// Referenced type kept for plugin authors' reference (payload shape of
// statusBar.* contributions).
export type { StatusbarItem }

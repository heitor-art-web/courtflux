"use client";

import Image from "next/image";
import Link from "next/link";
import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import styles from "./page.module.css";

type RosterPlayer = {
  id: string;
  name: string;
  jerseyNumber: number | null;
  position: string | null;
};

type Team = {
  id: string;
  name: string;
  shortName: string | null;
  season: string | null;
  ageGroup: string | null;
  isArchived: boolean;
  roster: RosterPlayer[];
};

type Props = {
  workspaceName: string;
  role: string;
  primaryTeamId: string | null;
  initialTeams: Team[];
};

type EditablePlayer = RosterPlayer & { localId: string };

function editableRoster(players: RosterPlayer[]): EditablePlayer[] {
  return players.map((player) => ({ ...player, localId: player.id }));
}

export function TeamManager({ workspaceName, role, primaryTeamId, initialTeams }: Props) {
  const router = useRouter();
  const canManage = ["owner", "admin", "analyst"].includes(role);
  const canChangePrimary = ["owner", "admin"].includes(role);
  const defaultTeam = initialTeams.find((team) => team.id === primaryTeamId)
    ?? initialTeams.find((team) => !team.isArchived)
    ?? initialTeams[0]
    ?? null;
  const [selectedId, setSelectedId] = useState(defaultTeam?.id ?? "");
  const [showArchived, setShowArchived] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const [newName, setNewName] = useState("");
  const [newShortName, setNewShortName] = useState("");
  const [newSeason, setNewSeason] = useState("");
  const [newAgeGroup, setNewAgeGroup] = useState("");

  const [name, setName] = useState(defaultTeam?.name ?? "");
  const [shortName, setShortName] = useState(defaultTeam?.shortName ?? "");
  const [season, setSeason] = useState(defaultTeam?.season ?? "");
  const [ageGroup, setAgeGroup] = useState(defaultTeam?.ageGroup ?? "");
  const [roster, setRoster] = useState<EditablePlayer[]>(defaultTeam ? editableRoster(defaultTeam.roster) : []);

  const selected = initialTeams.find((team) => team.id === selectedId) ?? null;
  const visibleTeams = useMemo(
    () => initialTeams.filter((team) => showArchived || !team.isArchived),
    [initialTeams, showArchived],
  );

  function selectTeam(team: Team) {
    setSelectedId(team.id);
    setName(team.name);
    setShortName(team.shortName ?? "");
    setSeason(team.season ?? "");
    setAgeGroup(team.ageGroup ?? "");
    setRoster(editableRoster(team.roster));
    setMessage("");
    setError("");
  }

  async function request(url: string, init: RequestInit) {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const response = await fetch(url, init);
      const data = (await response.json()) as { error?: string; id?: string; ok?: boolean };
      if (!response.ok) throw new Error(data.error || "CourtFlux could not save this change");
      router.refresh();
      return data;
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : String(requestError));
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function createTeam() {
    if (!newName.trim() || busy) return;
    const data = await request("/api/teams", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: newName,
        short_name: newShortName,
        season: newSeason,
        age_group: newAgeGroup,
      }),
    });
    if (data?.id) {
      setNewName("");
      setNewShortName("");
      setNewSeason("");
      setNewAgeGroup("");
      setMessage("Team created. Select it from the list to add a roster.");
    }
  }

  async function saveTeam() {
    if (!selected || !name.trim() || busy) return;
    const data = await request(`/api/teams/${selected.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        short_name: shortName,
        season,
        age_group: ageGroup,
      }),
    });
    if (data) setMessage("Team details saved.");
  }

  async function makePrimary() {
    if (!selected || !canChangePrimary || busy) return;
    const data = await request(`/api/teams/${selected.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ make_primary: true }),
    });
    if (data) setMessage("Primary team updated.");
  }

  async function toggleArchive() {
    if (!selected || busy) return;
    const nextArchived = !selected.isArchived;
    const data = await request(`/api/teams/${selected.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ is_archived: nextArchived }),
    });
    if (data) {
      setShowArchived(nextArchived ? true : showArchived);
      setMessage(nextArchived ? "Team archived. Historical analyses are unchanged." : "Team restored.");
    }
  }

  function addPlayer() {
    setRoster((current) => [
      ...current,
      { id: "", localId: crypto.randomUUID(), name: "", jerseyNumber: null, position: null },
    ]);
  }

  function updatePlayer(localId: string, field: "name" | "jerseyNumber" | "position", value: string) {
    setRoster((current) => current.map((player) => {
      if (player.localId !== localId) return player;
      if (field === "jerseyNumber") {
        return { ...player, jerseyNumber: value === "" ? null : Number(value) };
      }
      if (field === "name") return { ...player, name: value };
      return { ...player, position: value };
    }));
  }

  async function saveRoster() {
    if (!selected || busy) return;
    const data = await request(`/api/teams/${selected.id}/roster`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        players: roster.map((player) => ({
          id: player.id || undefined,
          name: player.name,
          jersey_number: player.jerseyNumber,
          position: player.position,
        })),
      }),
    });
    if (data) {
      // Reload once so newly inserted roster entries have their persistent IDs before another edit.
      window.location.reload();
    }
  }

  return (
    <main className={styles.shell}>
      <header className={styles.topbar}>
        <Link href="/" className={styles.brand} aria-label="CourtFlux home">
          <Image src="/brand/horizontal_logo_safe_crop.png" alt="CourtFlux — Basketball Intelligence" width={660} height={185} priority />
        </Link>
        <div className={styles.topActions}>
          <span>{workspaceName}</span>
          <Link href="/analyses">My analyses</Link>
          <Link href="/account">Dashboard</Link>
        </div>
      </header>

      <section className={styles.intro}>
        <div>
          <div className={styles.eyebrow}>Teams & rosters</div>
          <h1>Keep repeat team context ready without slowing down analysis.</h1>
          <p>Save the teams and current jersey context you use often. Rosters remain optional: CourtFlux still only needs two team names to start an analysis.</p>
        </div>
        <Link className={styles.primaryButton} href="/analysis/new">Start analysis</Link>
      </section>

      <div className={styles.layout}>
        <aside className={styles.teamSidebar}>
          <div className={styles.sidebarHeader}>
            <div><strong>Your teams</strong><span>{initialTeams.filter((team) => !team.isArchived).length} active</span></div>
            <label className={styles.archiveToggle}><input type="checkbox" checked={showArchived} onChange={(event) => setShowArchived(event.target.checked)} /> Show archived</label>
          </div>
          <div className={styles.teamList}>
            {visibleTeams.map((team) => (
              <button key={team.id} type="button" onClick={() => selectTeam(team)} className={team.id === selectedId ? styles.teamButtonActive : styles.teamButton}>
                <span><strong>{team.name}</strong><small>{[team.ageGroup, team.season].filter(Boolean).join(" · ") || "Basketball team"}</small></span>
                <span className={styles.teamMeta}>{team.id === primaryTeamId ? "Primary" : team.isArchived ? "Archived" : `${team.roster.length} players`}</span>
              </button>
            ))}
            {!visibleTeams.length ? <div className={styles.empty}>No teams to show yet.</div> : null}
          </div>

          {canManage ? (
            <details className={styles.createPanel} open={!initialTeams.length}>
              <summary>Add team</summary>
              <div className={styles.createFields}>
                <label>Team name *<input value={newName} onChange={(event) => setNewName(event.target.value)} placeholder="U18 Men" /></label>
                <label>Short name<input value={newShortName} onChange={(event) => setNewShortName(event.target.value)} placeholder="U18" /></label>
                <label>Age group<input value={newAgeGroup} onChange={(event) => setNewAgeGroup(event.target.value)} placeholder="U18" /></label>
                <label>Season<input value={newSeason} onChange={(event) => setNewSeason(event.target.value)} placeholder="2026/27" /></label>
                <button className={styles.secondaryButton} type="button" disabled={!newName.trim() || busy} onClick={() => void createTeam()}>Create team</button>
              </div>
            </details>
          ) : null}
        </aside>

        <section className={styles.workspace}>
          {selected ? (
            <>
              <section className={styles.panel}>
                <div className={styles.panelHeader}>
                  <div>
                    <div className={styles.eyebrow}>{selected.isArchived ? "Archived team" : selected.id === primaryTeamId ? "Primary team" : "Team details"}</div>
                    <h2>{selected.name}</h2>
                    <p>These fields help you recognise the team across seasons. They do not block analysis.</p>
                  </div>
                  <div className={styles.panelActions}>
                    {canChangePrimary && !selected.isArchived && selected.id !== primaryTeamId ? <button type="button" className={styles.secondaryButton} onClick={() => void makePrimary()} disabled={busy}>Make primary</button> : null}
                    {canManage ? <button type="button" className={styles.textButton} onClick={() => void toggleArchive()} disabled={busy}>{selected.isArchived ? "Restore team" : "Archive team"}</button> : null}
                  </div>
                </div>

                <div className={styles.formGrid}>
                  <label>Team name<input value={name} onChange={(event) => setName(event.target.value)} disabled={!canManage || busy} /></label>
                  <label>Short name<input value={shortName} onChange={(event) => setShortName(event.target.value)} disabled={!canManage || busy} /></label>
                  <label>Age group<input value={ageGroup} onChange={(event) => setAgeGroup(event.target.value)} placeholder="U16, U18, Senior…" disabled={!canManage || busy} /></label>
                  <label>Season<input value={season} onChange={(event) => setSeason(event.target.value)} placeholder="2026/27" disabled={!canManage || busy} /></label>
                </div>
                {canManage ? <button className={styles.primaryButton} type="button" onClick={() => void saveTeam()} disabled={!name.trim() || busy}>Save team details</button> : null}
              </section>

              <section className={styles.panel}>
                <div className={styles.panelHeader}>
                  <div>
                    <div className={styles.eyebrow}>Current roster</div>
                    <h2>Jersey context for future analyses</h2>
                    <p>Player names and jersey numbers can prefill New Analysis. Editing this roster never rewrites historical analysis snapshots.</p>
                  </div>
                  {canManage ? <button type="button" className={styles.secondaryButton} onClick={addPlayer} disabled={busy || roster.length >= 60}>Add player</button> : null}
                </div>

                <div className={styles.rosterHeader} aria-hidden="true"><span>#</span><span>Player</span><span>Position</span><span /></div>
                <div className={styles.rosterList}>
                  {roster.map((player) => (
                    <div className={styles.rosterRow} key={player.localId}>
                      <input aria-label={`Jersey number for ${player.name || "player"}`} type="number" min="0" max="999" value={player.jerseyNumber ?? ""} onChange={(event) => updatePlayer(player.localId, "jerseyNumber", event.target.value)} disabled={!canManage || busy} placeholder="#" />
                      <input aria-label="Player name" value={player.name} onChange={(event) => updatePlayer(player.localId, "name", event.target.value)} disabled={!canManage || busy} placeholder="Player name" />
                      <input aria-label={`Position for ${player.name || "player"}`} value={player.position ?? ""} onChange={(event) => updatePlayer(player.localId, "position", event.target.value)} disabled={!canManage || busy} placeholder="Optional" />
                      {canManage ? <button type="button" className={styles.removeButton} onClick={() => setRoster((current) => current.filter((item) => item.localId !== player.localId))} disabled={busy} aria-label={`Remove ${player.name || "player"} from current roster`}>Remove</button> : <span />}
                    </div>
                  ))}
                  {!roster.length ? <div className={styles.empty}><strong>No saved roster yet.</strong><span>You can leave it empty and analyse anonymous player tracks, or add the jersey context you already know.</span></div> : null}
                </div>
                {canManage ? <button className={styles.primaryButton} type="button" onClick={() => void saveRoster()} disabled={busy}>Save roster</button> : null}
              </section>
            </>
          ) : (
            <section className={styles.panel}>
              <div className={styles.empty}><strong>Create your first team.</strong><span>Once saved, you can add an optional roster and reuse it in New Analysis.</span></div>
            </section>
          )}

          {error ? <div className={styles.error} role="alert">{error}</div> : null}
          {message ? <div className={styles.success} role="status">{message}</div> : null}
        </section>
      </div>
    </main>
  );
}
